#!/usr/bin/env python
"""
Exp10: Polytope Loss training script.
Based on exp8b train.py with attention entropy penalty added.

Usage:
    uv run python exp10_polytope/train_polytope.py <lang> <arm> <lambda>

    lang:   en | fr
    arm:    fertility | wals
    lambda: float (e.g., 1.65)

Example:
    uv run python exp10_polytope/train_polytope.py en fertility 1.65
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import csv
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm
import sys
import json
import signal
from datetime import datetime
from safetensors.torch import save_file, load_file


# ==================== SEED ====================
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ==================== DEVICE ====================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# ==================== CONFIG ====================
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
ARM = sys.argv[2] if len(sys.argv) > 2 else "fertility"
POLYTOPE_LAMBDA = float(sys.argv[3]) if len(sys.argv) > 3 else 1.65

MODEL_SIZE = "125M"
TARGET_STEPS = 100_000

cfg = dict(n_layers=12, d_model=768, n_heads=12, d_ff=3072, batch=16, seq=512)
BATCH = cfg["batch"]
SEQ = cfg["seq"]

BASE_PATH = Path(os.environ.get("DATA_PATH", "/workspace/data"))
RUN_NAME = f"{LANG}_{ARM}_lambda{POLYTOPE_LAMBDA:.2f}"
CHECKPOINTS = BASE_PATH / "checkpoints" / "exp10" / RUN_NAME
CHECKPOINTS.mkdir(parents=True, exist_ok=True)
CHUNK_DIR = BASE_PATH / "chunks"
TOKENIZER_PATH = BASE_PATH / "joint_tokenizer.json"

TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
VOCAB = TOKENIZER.get_vocab_size()

print(f"=== Exp10: Polytope Loss ===")
print(f"Language: {LANG.upper()}, Arm: {ARM}, Lambda: {POLYTOPE_LAMBDA:.2f}")
print(f"Run name: {RUN_NAME}")
print(f"Config: {MODEL_SIZE}, batch={BATCH}, seq={SEQ}, vocab={VOCAB}")
print(f"Target: {TARGET_STEPS:,} steps (~{TARGET_STEPS * BATCH * SEQ / 1e6:.0f}M tokens)")


# ==================== MODEL ====================
class TransformerBlock(nn.Module):
    """Transformer block that returns attention weights for Polytope Loss."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attn=False):
        ln_x = self.ln1(x)
        T = ln_x.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=ln_x.device), diagonal=1).bool()
        attn_out, attn_weights = self.attn(
            ln_x, ln_x, ln_x,
            attn_mask=causal_mask,
            need_weights=return_attn,
            average_attn_weights=False,  # Get per-head weights: (B, n_heads, T, T)
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        if return_attn:
            return x, attn_weights
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, return_attn=False):
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.drop(self.embed(x) + self.pos_embed(pos))

        all_attn = []
        for block in self.blocks:
            if return_attn:
                x, attn_w = block(x, return_attn=True)
                all_attn.append(attn_w)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        if return_attn:
            return logits, all_attn
        return logits


# ==================== POLYTOPE LOSS ====================
def attention_entropy(attn_weights_list):
    """
    Compute mean Shannon entropy across all attention heads and layers.

    Args:
        attn_weights_list: list of (B, n_heads, T, T) tensors, one per layer

    Returns:
        Scalar mean entropy
    """
    total_entropy = 0.0
    count = 0
    for attn_w in attn_weights_list:
        # attn_w: (B, n_heads, T, T) — already softmaxed by nn.MultiheadAttention
        # Clamp for numerical stability
        attn_w = attn_w.clamp(min=1e-9)
        entropy = -torch.sum(attn_w * torch.log(attn_w), dim=-1)  # (B, n_heads, T)
        total_entropy += entropy.mean()
        count += 1
    return total_entropy / count


def polytope_loss(ce_loss, attn_weights_list, polytope_lambda):
    """
    Polytope Loss: CE + lambda * mean_attention_entropy

    Operationalizes the hypothesis that morphologically rich languages
    implicitly regularize attention. This loss applies explicit regularization
    for languages that lack morphological signal.
    """
    attn_entropy = attention_entropy(attn_weights_list)
    return ce_loss + polytope_lambda * attn_entropy, attn_entropy


# ==================== PROBES ====================
TEMPERATURE = 0.1

GRAMMAR_PROBES = {
    "en": [
        {"prompt": "The cat ", "good": ["is", "was", "sits", "runs"], "bad": ["are", "were", "sit", "run"], "category": "sv_agreement"},
        {"prompt": "The dog ", "good": ["is", "was", "barks", "runs"], "bad": ["are", "were", "bark", "run"], "category": "sv_agreement"},
        {"prompt": "She ", "good": ["is", "was", "has", "does"], "bad": ["are", "were", "have", "do"], "category": "sv_agreement"},
        {"prompt": "The cats ", "good": ["are", "were", "sit", "run"], "bad": ["is", "was", "sits", "runs"], "category": "sv_agreement"},
        {"prompt": "They ", "good": ["are", "were", "have", "do"], "bad": ["is", "was", "has", "does"], "category": "sv_agreement"},
        {"prompt": "I saw a ", "good": ["cat", "dog", "bird", "man"], "bad": ["apple", "elephant", "orange"], "category": "article"},
        {"prompt": "I saw an ", "good": ["apple", "elephant", "orange"], "bad": ["cat", "dog", "bird"], "category": "article"},
        {"prompt": "black and ", "good": ["white"], "bad": ["red", "green", "blue"], "category": "collocation"},
        {"prompt": "up and ", "good": ["down"], "bad": ["left", "right", "over"], "category": "collocation"},
        {"prompt": "day and ", "good": ["night"], "bad": ["morning", "evening", "time"], "category": "collocation"},
    ],
    "fr": [
        {"prompt": "Le chat ", "good": ["est", "était", "noir", "petit"], "bad": ["sont", "étaient", "noire", "petite"], "category": "gender"},
        {"prompt": "La maison ", "good": ["est", "était", "grande", "belle"], "bad": ["sont", "étaient", "grand", "beau"], "category": "gender"},
        {"prompt": "Il ", "good": ["est", "était", "a", "fait"], "bad": ["sont", "étaient", "ont", "font"], "category": "gender"},
        {"prompt": "Elle ", "good": ["est", "était", "a", "fait"], "bad": ["sont", "étaient", "ont", "font"], "category": "gender"},
        {"prompt": "Les chats ", "good": ["sont", "étaient", "noirs"], "bad": ["est", "était", "noir"], "category": "number"},
        {"prompt": "Les maisons ", "good": ["sont", "étaient", "grandes"], "bad": ["est", "était", "grande"], "category": "number"},
        {"prompt": "Je vois le ", "good": ["chat", "chien", "livre"], "bad": ["maison", "femme", "table"], "category": "article_gender"},
        {"prompt": "Je vois la ", "good": ["maison", "femme", "table"], "bad": ["chat", "chien", "livre"], "category": "article_gender"},
        {"prompt": "noir et ", "good": ["blanc"], "bad": ["rouge", "vert", "bleu"], "category": "collocation"},
        {"prompt": "jour et ", "good": ["nuit"], "bad": ["matin", "soir", "temps"], "category": "collocation"},
    ],
}

EMERGENCE_PROBES = {
    "en": [
        {"prompt": "2 + 2 =", "good": ["4"], "bad": ["3", "5", "6"], "category": "arithmetic"},
        {"prompt": "5 + 3 =", "good": ["8"], "bad": ["7", "9", "6"], "category": "arithmetic"},
        {"prompt": "10 - 4 =", "good": ["6"], "bad": ["5", "7", "14"], "category": "arithmetic"},
        {"prompt": "The capital of France is", "good": ["Paris"], "bad": ["London", "Berlin", "Rome"], "category": "knowledge"},
        {"prompt": "The sun rises in the", "good": ["east"], "bad": ["west", "north", "south"], "category": "knowledge"},
        {"prompt": "The opposite of hot is", "good": ["cold"], "bad": ["warm", "heat", "fire"], "category": "semantic"},
        {"prompt": "The opposite of big is", "good": ["small"], "bad": ["large", "huge", "giant"], "category": "semantic"},
    ],
    "fr": [
        {"prompt": "2 + 2 =", "good": ["4"], "bad": ["3", "5", "6"], "category": "arithmetic"},
        {"prompt": "5 + 3 =", "good": ["8"], "bad": ["7", "9", "6"], "category": "arithmetic"},
        {"prompt": "La capitale de la France est", "good": ["Paris"], "bad": ["Londres", "Berlin", "Rome"], "category": "knowledge"},
        {"prompt": "Le contraire de chaud est", "good": ["froid"], "bad": ["tiède", "chaleur", "feu"], "category": "semantic"},
        {"prompt": "Le contraire de grand est", "good": ["petit"], "bad": ["large", "énorme", "gros"], "category": "semantic"},
    ],
}


def get_token_probability(model, tokenizer, prompt, target, device):
    prompt_ids = tokenizer.encode(prompt).ids
    target_ids = tokenizer.encode(" " + target).ids
    if not target_ids:
        target_ids = tokenizer.encode(target).ids
    if not target_ids:
        return 0.0
    target_id = target_ids[0]
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x, return_attn=False)
        probs = F.softmax(logits[0, -1, :] / TEMPERATURE, dim=-1)
        return probs[target_id].item()


def evaluate_probe(model, tokenizer, probe, device):
    prompt = probe["prompt"]
    good_probs = [get_token_probability(model, tokenizer, prompt, w, device) for w in probe["good"]]
    bad_probs = [get_token_probability(model, tokenizer, prompt, w, device) for w in probe["bad"]]
    avg_good = np.mean(good_probs)
    avg_bad = np.mean(bad_probs)
    log_ratio = np.log(avg_good + 1e-10) - np.log(avg_bad + 1e-10)
    correct = max(good_probs) > max(bad_probs)
    return {"prompt": prompt, "category": probe["category"], "avg_good_prob": avg_good, "avg_bad_prob": avg_bad, "log_ratio": log_ratio, "correct": correct}


def run_grammar_probes(model, tokenizer, lang, step, device):
    model.eval()
    results = [evaluate_probe(model, tokenizer, p, device) for p in GRAMMAR_PROBES[lang]]
    total_correct = sum(1 for r in results if r["correct"])
    accuracy = 100 * total_correct / len(results)
    mean_log_ratio = np.mean([r["log_ratio"] for r in results])
    model.train()
    return {"step": step, "accuracy": accuracy, "mean_log_ratio": mean_log_ratio, "correct": total_correct, "total": len(results)}


def run_emergence_probes(model, tokenizer, lang, step, device):
    model.eval()
    results = [evaluate_probe(model, tokenizer, p, device) for p in EMERGENCE_PROBES[lang]]
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0, "log_ratios": []}
        categories[cat]["total"] += 1
        if r["correct"]:
            categories[cat]["correct"] += 1
        categories[cat]["log_ratios"].append(r["log_ratio"])
    model.train()
    return {"step": step, "categories": categories}


# ==================== LOGGING ====================
def setup_logging(base_path, run_name):
    log_dir = base_path / "logs" / "exp10" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def log_training_step(step, ce_loss, attn_entropy, total_loss, ppl, log_dir, run_name):
    log_file = log_dir / "training.csv"
    file_exists = log_file.exists()
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["step", "ce_loss", "attn_entropy", "total_loss", "perplexity", "timestamp"])
        writer.writerow([step, f"{ce_loss:.4f}", f"{attn_entropy:.4f}", f"{total_loss:.4f}", f"{ppl:.2f}", datetime.now().isoformat()])


def log_grammar_results(results, log_dir):
    log_file = log_dir / "grammar_probes.csv"
    file_exists = log_file.exists()
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["step", "accuracy", "mean_log_ratio", "correct", "total", "timestamp"])
        writer.writerow([results["step"], f"{results['accuracy']:.2f}", f"{results['mean_log_ratio']:.4f}", results["correct"], results["total"], datetime.now().isoformat()])


def log_emergence_results(results, log_dir):
    log_file = log_dir / "emergence_probes.csv"
    file_exists = log_file.exists()
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["step", "category", "accuracy", "mean_log_ratio", "timestamp"])
        for cat, data in results["categories"].items():
            acc = 100 * data["correct"] / data["total"]
            log_ratio = np.mean(data["log_ratios"])
            writer.writerow([results["step"], cat, f"{acc:.2f}", f"{log_ratio:.4f}", datetime.now().isoformat()])


# ==================== DATA ====================
def data_stream():
    chunks = sorted(CHUNK_DIR.glob(f"{LANG}_chunk_*.npy"))
    print(f"Found {len(chunks)} chunks for {LANG}")
    if len(chunks) == 0:
        raise ValueError(f"No chunks found in {CHUNK_DIR} for {LANG}")
    while True:
        rng = np.random.RandomState(RANDOM_SEED)
        chunk_order = list(range(len(chunks)))
        rng.shuffle(chunk_order)
        for chunk_idx in chunk_order:
            tokens = np.load(chunks[chunk_idx], mmap_mode='r')
            for i in range(0, len(tokens) - SEQ - 1, SEQ // 2):
                yield tokens[i:i + SEQ + 1]


# ==================== CHECKPOINTING ====================
def save_checkpoint(step, model, optimizer, ce_loss, attn_entropy, total_loss):
    checkpoint_path = CHECKPOINTS / f"checkpoint_{step}.json"
    model_path = CHECKPOINTS / f"model_{step}.safetensors"
    optimizer_path = CHECKPOINTS / f"optimizer_{step}.pt"

    meta = {
        "step": step,
        "ce_loss": float(ce_loss),
        "attn_entropy": float(attn_entropy),
        "total_loss": float(total_loss),
        "polytope_lambda": POLYTOPE_LAMBDA,
        "arm": ARM,
        "lang": LANG,
        "model_size": MODEL_SIZE,
        "run_name": RUN_NAME,
        "timestamp": datetime.now().isoformat(),
        "config": cfg,
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(meta, f, indent=2)

    state = {k: v.clone() for k, v in model.state_dict().items()}
    save_file(state, str(model_path))
    torch.save(optimizer.state_dict(), str(optimizer_path))
    print(f"Checkpoint saved: step {step}, ce={ce_loss:.4f}, entropy={attn_entropy:.4f}, total={total_loss:.4f}")


def load_checkpoint(model, optimizer):
    checkpoints = list(CHECKPOINTS.glob("checkpoint_*.json"))
    if not checkpoints:
        return 0
    latest = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))
    step = int(latest.stem.split('_')[1])
    model_path = CHECKPOINTS / f"model_{step}.safetensors"
    optimizer_path = CHECKPOINTS / f"optimizer_{step}.pt"
    if model_path.exists():
        state_dict = load_file(str(model_path))
        model.load_state_dict(state_dict)
        print(f"Loaded model from step {step}")
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(str(optimizer_path), weights_only=True))
        print(f"Loaded optimizer from step {step}")
    return step


# ==================== TRAINING ====================
def main():
    # Save experiment config
    config_path = CHECKPOINTS / "experiment_config.json"
    experiment_config = {
        "experiment": "exp10_polytope",
        "lang": LANG,
        "arm": ARM,
        "polytope_lambda": POLYTOPE_LAMBDA,
        "model_size": MODEL_SIZE,
        "target_steps": TARGET_STEPS,
        "batch_size": BATCH,
        "seq_length": SEQ,
        "seed": RANDOM_SEED,
        "run_name": RUN_NAME,
        "started": datetime.now().isoformat(),
    }
    with open(config_path, 'w') as f:
        json.dump(experiment_config, f, indent=2)

    model = Transformer(
        vocab_size=VOCAB, d_model=cfg["d_model"], n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"]
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.01)
    start_step = load_checkpoint(model, optimizer)
    start_step += 1

    if start_step > TARGET_STEPS:
        print(f"Training already complete (step {start_step - 1} >= {TARGET_STEPS})")
        return

    log_dir = setup_logging(BASE_PATH, RUN_NAME)
    stream = data_stream()

    shutdown_requested = False
    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        print("\nShutdown requested, finishing current step...")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    model.train()
    pbar = tqdm(range(start_step, TARGET_STEPS + 1), desc=f"Exp10 {RUN_NAME}", initial=start_step - 1, total=TARGET_STEPS)

    running_ce = 0.0
    running_entropy = 0.0
    running_total = 0.0
    log_interval = 50
    checkpoint_interval = 1000

    for step in pbar:
        if shutdown_requested:
            save_checkpoint(step - 1, model, optimizer, running_ce / log_interval, running_entropy / log_interval, running_total / log_interval)
            break

        batch_x, batch_y = [], []
        for _ in range(BATCH):
            seq = next(stream)
            batch_x.append(seq[:-1])
            batch_y.append(seq[1:])

        x = torch.tensor(np.array(batch_x), dtype=torch.long, device=DEVICE)
        y = torch.tensor(np.array(batch_y), dtype=torch.long, device=DEVICE)

        logits, attn_weights_list = model(x, return_attn=True)
        ce_loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))
        total_loss, attn_entropy_val = polytope_loss(ce_loss, attn_weights_list, POLYTOPE_LAMBDA)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_ce += ce_loss.item()
        running_entropy += attn_entropy_val.item()
        running_total += total_loss.item()

        if step % log_interval == 0:
            avg_ce = running_ce / log_interval
            avg_entropy = running_entropy / log_interval
            avg_total = running_total / log_interval
            ppl = np.exp(avg_ce)
            log_training_step(step, avg_ce, avg_entropy, avg_total, ppl, log_dir, RUN_NAME)
            pbar.set_postfix({
                'ce': f'{avg_ce:.3f}',
                'H': f'{avg_entropy:.3f}',
                'ppl': f'{ppl:.1f}',
                'λ': f'{POLYTOPE_LAMBDA:.2f}',
            })
            running_ce = 0.0
            running_entropy = 0.0
            running_total = 0.0

        if step % checkpoint_interval == 0:
            save_checkpoint(step, model, optimizer, ce_loss.item(), attn_entropy_val.item(), total_loss.item())

            grammar_results = run_grammar_probes(model, TOKENIZER, LANG, step, DEVICE)
            log_grammar_results(grammar_results, log_dir)
            print(f"  Grammar: {grammar_results['accuracy']:.1f}% acc, log_ratio={grammar_results['mean_log_ratio']:.2f}")

            emergence_results = run_emergence_probes(model, TOKENIZER, LANG, step, DEVICE)
            log_emergence_results(emergence_results, log_dir)
            print(f"  Emergence probes logged")

    print(f"\nTraining complete: {RUN_NAME}")


if __name__ == "__main__":
    main()
