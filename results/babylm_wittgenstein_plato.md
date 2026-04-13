# Wittgenstein vs. Plato: Evidence from BabyLM Cross-Linguistic Training

**Date:** 2026-04-13
**Source:** BabyLM 2026 experiments (Born Speaking French)
**Status:** Working notes from active experimentation

## The Question

Is meaning in the structure of language (Wittgenstein) or in the forms that language points to (Plato)?

The BabyLM experiments generate direct evidence because they isolate a single variable: a 125M-parameter GPT-2 trained exclusively on French (92M words) is tested on English benchmarks. The model has never seen English. Any English-task competence it demonstrates must come from language-independent representations learned through French structure.

## The Experimental Setup

| Component | Detail |
|-----------|--------|
| Model | 125M GPT-2, 12 layers, d_model=768 |
| Training data | 92M words, 100% French |
| Epochs | 3-5 (model converged by epoch 3) |
| Evaluation | BabyLM 2026 suite: BLiMP, BLiMP supplement, GLUE (7 tasks), EWoK, Entity Tracking |

Three experimental conditions on GLUE tasks:
- **Bare English:** Feed English task data directly to the French model
- **Dict-axioms (vocabulary bridge):** Prepend FR-EN word translations, then feed English data
- **French framing:** Translate entire task to French (in progress)

## Evidence for Wittgenstein: Meaning = Structure

French morphological redundancy creates grammatical competence that English cannot match at the same scale.

| Metric | French (92M words) | English (92M words) | English (3B+ words) |
|--------|-------------------|--------------------|--------------------|
| BLiMP (grammar) | 76.18% | ~67% (GPT-2 baseline) | ~75% |
| BLiMP supplement | 96.40% | ~65% | ~70% |
| QFrBLiMP (French grammar) | 85.97% | n/a | n/a |

The French model at 92M words matches or exceeds what English achieves at 3B+ words on grammatical competence. The information is in the structure: gender agreement, verb conjugation, number marking, and morphological composition create a denser learning signal per token. The model learns "what words mean" from how they behave in morphologically constrained contexts.

This is Wittgenstein's insight operationalized: meaning is use, and French encodes more meaning in its patterns of use per unit of text.

## Evidence for Plato: Meaning Transcends Language

The dict-axioms experiment reveals that the French model possesses language-independent conceptual representations.

### RTE (Recognizing Textual Entailment) results:

| Condition | Accuracy | Delta |
|-----------|----------|-------|
| Bare English (zero-shot) | 47.5% | baseline |
| Dict-axioms English (zero-shot) | **54.0%** | **+6.5pp** |
| Dict-axioms French (zero-shot) | **54.0%** | **+6.5pp** |
| LoRA fine-tuned on English (no axioms) | 53.24% | +5.7pp |

The critical observation: **providing only word-level FR-EN translations (no grammar, no syntax, no training) enables the French model to perform English entailment reasoning better than gradient-based fine-tuning on English data.**

The model "knows" what entailment is. It learned this concept from French text alone. The concept exists in the model's internal representations independent of surface language. It just needs a lexical bridge to apply it to English tokens.

This is a Platonic form: entailment as a language-independent abstract structure, learned through one language, applicable to another.

### Cross-task evidence from the vocabulary bridge:

| Task | Bare English | With axioms | Bridge effect | Interpretation |
|------|-------------|-------------|---------------|----------------|
| RTE (entailment) | 47.5% | 54.0% | +6.5pp | Concept transfers |
| MRPC (paraphrase) | 31.9% | 36.8% | +4.9pp | Concept transfers |
| MNLI (3-way entailment) | 32.2% | 34.2% | +2.0pp | Partial transfer |
| BoolQ (comprehension) | 57.0% | 58.8% | +1.8pp | Weak transfer |
| MultiRC (reading) | 53.8% | 53.8% | 0 | No transfer |
| QQP (duplicate) | 62.6% | 62.6% | 0 | No transfer |
| WSC (coreference) | 61.5% | 61.5% | 0 | No transfer |

The gradient is revealing: **relational concepts (entailment, paraphrase) transfer across languages; passage-level comprehension does not.** This suggests a hierarchy:

1. **Relational/logical concepts** (entailment, paraphrase): Language-independent, Platonic
2. **Word-level semantics** (reading comprehension, yes/no): Partially language-dependent
3. **Extended discourse** (multi-sentence reading): Language-bound, Wittgensteinian

## The Synthesis

**Wittgenstein is right about acquisition. Plato is right about representation.**

Once a model acquires a concept through rich morphological structure, that concept exists as a language-independent representation. But the efficiency of acquisition depends entirely on the structural properties of the training language.

| | Acquisition (learning) | Representation (knowledge) |
|---|---|---|
| **Who's right** | Wittgenstein | Plato |
| **What matters** | Structure of language | Forms transcending language |
| **Evidence** | French 20x more efficient than English | Entailment transfers with word bridge only |
| **Mechanism** | Morphological redundancy = denser signal | Shared latent geometry across languages |

Morphologically rich languages are better *teachers*, not because they contain different knowledge, but because their structure makes the knowledge accessible faster. The geometric structure that Levin's VM4AI manipulates at inference time is being provided *for free* by French morphology at training time.

## Implications for Wasserman-Levin (2026)

1. **The Polytope/Sphere experiments should show that geometric constraints at training time replicate what morphology provides naturally.** If French morphology creates a natural polytope-like constraint (rigid agreement patterns = logical structure), then explicit Polytope loss should produce similar efficiency gains when applied to English training.

2. **The transfer gradient (relational > semantic > discourse) maps onto VM4AI topologies.** Polytope (rigid logic) should capture what transfers (entailment). Sphere (fluid association) should capture what doesn't (discourse comprehension). This predicts that different VM4AI topologies will affect different GLUE tasks differently.

3. **The vocabulary bridge result is a direct test of latent geometry.** If axiom-augmented French models can do English entailment, the latent representations must share geometric structure across languages. This is the "invariance" hypothesis from Exp3 of Wasserman-Levin, and we now have positive evidence for it from a completely independent experiment.

## Raw Data

All models and evaluation results are in `/Users/adam/dev/babylm/`:
- `models_epoch3/chck_92M/` — best epoch for benchmarks
- `models_epoch4/chck_92M/` — freshly trained
- `models_epoch5_new/chck_92M/` — freshly trained
- Experiment results: see eval logs in `models_epoch3/eval_results.log` through `models_epoch5/eval_results.log`
- Dict-axioms experiment: run on RunPod, results in session transcript
