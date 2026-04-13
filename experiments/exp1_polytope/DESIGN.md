# Exp1: Polytope Loss

## Can Attention Entropy Minimization Synthesize Morphological Advantage?

## Pre-registration

**OSF registration**: [TODO: register before first training run]

## 1. Motivation

Two independent findings motivate this experiment:

**Cross-linguistic training dynamics (Wasserman, 2026)**: Controlled ablation experiments across 12 languages established that morphologically rich languages reach grammatical competence faster and achieve lower perplexity than morphologically poor languages. The strongest predictors are WALS VerbSynth (r=-0.88) and Agreement (r=-0.78). English grammar remains at chance level despite extensive training.

**Geometric cognitive topologies (Levin & Levin, 2025)**: The VM4AI framework demonstrates that imposing geometric structure on AI reasoning produces measurably different cognitive outputs. The Polytope topology — rigid, hard-edged, no wiggle room — enforces precise logical reasoning at inference time. VM4AI's Topology Quantizer tool translates this geometric concept into a training-time loss function: the Polytope Loss, which minimizes attention entropy to force focused structural processing.

The hypothesis connecting these findings: if morphological redundancy works by constraining attention to focus on structurally relevant tokens (as the cross-linguistic data suggests), then the Polytope's rigid geometric constraint — applied as an explicit attention entropy penalty during training — might replicate this effect for languages that lack morphological signal.

This experiment tests whether the Polytope Loss can break the English grammar ceiling observed in exp8b.

## 2. Background

### Exp8b baselines (same tokenizer, same architecture)

| Language | Grammar (final) | Tokens to 60% | Val PPL | Tokens |
|----------|----------------|---------------|---------|--------|
| English  | 87%*           | 22.5M         | 74.4    | 780M   |
| French   | 87%            | 6.1M          | 37.7    | 780M   |

*Note: Exp8b English reached 87% grammar in the balanced multilingual experiment. In the original exp1 (English-only, batch=2), English grammar was stuck at 40%. The discrepancy may reflect batch size differences (exp8b used batch=16 vs exp1 batch=2). We use the exp8b results as our baseline since this experiment uses the same tokenizer and batch configuration.

### The intervention

Add a weighted attention entropy penalty to the standard cross-entropy loss:

```
total_loss = CE(logits, labels) + lambda * mean(H(attention_weights))
```

Where H is the Shannon entropy of the attention distribution across all heads and layers.

### Two parameterizations of lambda

**Arm 1, BPE Fertility (Edward Levin / VM4AI):**
Lambda scaled by the ratio of BPE tokens to whitespace-delimited words. Languages that expand more under BPE tokenization (higher fertility) receive higher lambda, on the theory that tokenization fragmentation disrupts morphological signal and needs stronger correction.

**Arm 2, WALS Composite:**
Lambda scaled by morphological features from the World Atlas of Language Structures. Uses the composite score (22A VerbSynth + 29A Agreement + 21B TAM + 20A Fusion) identified in exp8b as the strongest predictor of training efficiency. This has direct theoretical grounding: the features that predict morphological advantage should also predict how much regularization is needed to simulate it.

## 2.5 New evidence: BabyLM cross-linguistic transfer (Wasserman, 2026, unpublished)

Concurrent experiments for the BabyLM 2026 submission ("Born Speaking French") provide new evidence that sharpens predictions for this experiment.

A 125M GPT-2 trained exclusively on French (92M words) was tested on English benchmarks using a "dict-axioms" vocabulary bridge: simple FR-EN word translations prepended to the prompt, with no grammar, no syntax, and no fine-tuning. Results reveal a **transfer gradient** across GLUE tasks:

| Transfer tier | Tasks | Bridge effect | Character |
|---|---|---|---|
| **Relational/logical** | RTE (+6.5pp), MRPC (+4.9pp) | Strong | Rigid, structural |
| **Semantic** | MNLI (+2.0pp), BoolQ (+1.8pp) | Weak | Mixed |
| **Discourse** | MultiRC (0), QQP (0), WSC (0) | None | Fluid, contextual |

The critical finding: relational concepts (entailment, paraphrase) transfer cross-linguistically with only a lexical bridge, while discourse-level comprehension does not. This gradient maps directly onto VM4AI's topology distinction: Polytope (rigid/logic) corresponds to the tier that transfers; Sphere (fluid/creative) corresponds to the tier that does not.

This evidence motivates a new directional prediction (H4) not present in the original design.

## 3. Hypotheses

### H0 (null, from orthogonality finding)
Neither parameterization breaks the grammar ceiling. English grammar accuracy under Polytope Loss will remain within measurement noise of the exp8b baseline (<=50% on grammar probes) at all lambda values tested. Perplexity may improve.

**Basis**: Exp1 showed English PPL improved from 1340 to 777 (42% reduction) while grammar remained locked at 40%. If the grammar deficit is structural (absent from the data), no loss function can recover it.

### H1 (alternative)
At least one parameterization produces English grammar accuracy >50% sustained over 3 or more consecutive checkpoints (3000+ steps).

### H2 (conditional on H1)
If grammar improvement occurs, WALS-derived lambda produces equal or greater improvement than fertility-derived lambda, since WALS features are stronger predictors of training efficiency (r=-0.88) than tokenization-based proxies.

### H3 (French control)
French grammar accuracy under Polytope Loss will not exceed its exp8b baseline (87%). Morphologically rich languages already provide the regularization the loss term attempts to simulate; adding it artificially is redundant at best and harmful at worst (cf. exp8b interleaved EN/FR finding where mixing languages degraded French).

### H4 (probe stratification, from BabyLM transfer gradient)
If Polytope Loss improves grammar (H1 supported), the improvement will be **non-uniform across probe types**. Probes testing relational/logical structure (agreement, binding, argument structure) will improve before and more than probes testing discourse-level or pragmatic phenomena (filler-gap, island effects, garden-path recovery).

**Basis**: The BabyLM dict-axioms experiment (Wasserman, 2026, unpublished) demonstrates that a French-trained model's relational/logical competence transfers cross-linguistically with only a vocabulary bridge, while discourse-level comprehension does not. If the Polytope topology corresponds to rigid/logical structure (as VM4AI posits), its training-time analogue should preferentially improve the same tier of linguistic competence that transfers cross-linguistically. This prediction is falsifiable: if Polytope Loss improves all probe types uniformly, the VM4AI topology-to-transfer mapping does not hold.

## 4. Experimental Design

### 4.1 Controlled variables (identical to exp8b)

- **Architecture**: 125M GPT-2 style (12 layers, d_model=768, 12 heads, d_ff=3072)
- **Tokenizer**: Exp8b joint BPE (50k vocab). MUST NOT be retrained.
- **Optimizer**: AdamW, lr=6e-4, weight_decay=0.01
- **Random seed**: 42
- **Sequence length**: 512
- **Batch size**: 16
- **Data**: Same chunks from exp8b (English, French)

### 4.2 Independent variable

The Polytope Loss lambda, parameterized two ways:

**Arm 1, BPE Fertility:**

Lambda values based on Edward Levin's observed sweet spot range.

| Run | Language | Lambda | Source |
|-----|----------|--------|--------|
| 1   | English  | 1.50   | Below sweet spot |
| 2   | English  | 1.65   | Sweet spot midpoint |
| 3   | English  | 1.85   | Above sweet spot |
| 7   | French   | 1.65   | Sweet spot (control) |

**Arm 2, WALS Composite:**

WALS composite scores from exp8b:
- English: VerbSynth=2, Agreement=4, TAM=unknown, Fusion=2 → composite ~8
- French: VerbSynth=4, Agreement=7, TAM=unknown, Fusion=2 → composite ~13

Lambda derived by normalizing WALS composite to the same [1.50, 1.85] range as Arm 1, allowing direct comparison:

```
lambda = 1.50 + (1.85 - 1.50) * (max_wals - lang_wals) / (max_wals - min_wals)
```

Note: HIGHER lambda for LOWER WALS scores (languages with less morphology need more regularization).

| Run | Language | WALS Composite | Lambda | Source |
|-----|----------|---------------|--------|--------|
| 4   | English  | 8             | 1.73   | WALS-derived midpoint |
| 5   | English  | 8             | 1.50   | WALS low (sensitivity) |
| 6   | English  | 8             | 1.85   | WALS high (sensitivity) |
| 8   | French   | 13            | 1.50   | WALS-derived (control) |

### 4.3 Dependent variables

Measured every 1000 steps:

1. **Grammar probe accuracy** (primary outcome): same probes as exp8b
2. **Validation perplexity**: held-out set from exp8b
3. **Mean attention entropy**: to verify the loss is actually affecting attention patterns
4. **Training loss decomposition**: CE component vs polytope component separately

### 4.4 Run length

100,000 steps (~200M tokens). Rationale: English grammar in exp8b was flat from early training through 780M tokens. If the Polytope Loss moves the needle, the effect should be visible well before 200M tokens. This is 4x the tokens at which French reached 60% grammar (6.1M → ~50M generous upper bound).

### 4.5 Success criteria

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| Grammar breakthrough | >50% for 3+ consecutive checkpoints | Exceeds chance + noise |
| Perplexity improvement | <74.4 (exp8b English baseline) | Faster convergence |
| French interference | French grammar <80% | Polytope Loss harms natural morphological signal |

### 4.6 Run matrix

| Run | Language | Arm | Lambda | Steps | Purpose |
|-----|----------|-----|--------|-------|---------|
| 1   | English  | Fertility | 1.50 | 100k | Below sweet spot |
| 2   | English  | Fertility | 1.65 | 100k | Sweet spot |
| 3   | English  | Fertility | 1.85 | 100k | Above sweet spot |
| 4   | English  | WALS | 1.50 | 100k | WALS low |
| 5   | English  | WALS | 1.73 | 100k | WALS midpoint |
| 6   | English  | WALS | 1.85 | 100k | WALS high |
| 7   | French   | Fertility | 1.65 | 100k | French control (fertility) |
| 8   | French   | WALS | 1.50 | 100k | French control (WALS) |

**Total**: 8 runs x 100k steps = 800k steps
**Estimated compute**: ~2 days on 2x RTX 4090

## 5. Analysis Plan

### Primary analysis
Compare grammar probe accuracy across all 8 runs against exp8b baselines. Binary outcome: did any run break the 50% threshold for 3+ consecutive checkpoints?

### Secondary analyses
1. **Arm comparison**: If H1 confirmed, compare Arm 1 vs Arm 2 grammar trajectories
2. **Lambda sensitivity**: Plot grammar accuracy vs lambda to identify any threshold effects
3. **Perplexity decomposition**: Did the polytope component reduce entropy without affecting grammar? (Would strengthen orthogonality finding)
4. **French control**: Any degradation relative to exp8b baseline?

### Reporting commitment
All results will be reported regardless of outcome. Null results are informative: they strengthen the Language-Only Hypothesis by showing that the grammar deficit cannot be overcome through training dynamics alone.

## 6. Collaboration

### Edward Levin ([VM4AI](https://vm4ai.com))
- Polytope Loss concept, derived from VM4AI's Polytope cognitive topology (Rigid/Logic: "a shape with hard edges, no wiggle room")
- Topology Quantizer tool for lambda visualization and code generation
- BPE fertility parameterization and lambda sweet-spot analysis (1.50-1.85 range)
- Core hypothesis: geometric training constraints can simulate morphological regularization

### Adam Wasserman ([fractal-language](https://github.com/adamzwasserman/fractal-language))
- Cross-linguistic baselines from 12-language controlled ablation (exp8b)
- WALS morphological parameterization (VerbSynth, Agreement, TAM, Fusion)
- Experimental design, pre-registration framework, and analysis plan
- Training infrastructure and compute

### Intellectual property
- VM4AI geometric engine and cognitive topologies: Edward Levin & Karen Levin, CC-BY-NC-SA 4.0
- Morphological calibration methods and per-language lambda tuning: Subject to provisional patents held by Adam Wasserman
- Joint experimental results: Shared with attribution to both researchers

## 7. Relation to Other Experiments

| Experiment | What it showed | How this experiment builds on it |
|------------|---------------|----------------------------------|
| Wasserman Exp1 | PPL and grammar are orthogonal in English | Tests if Polytope Loss breaks orthogonality |
| Wasserman Exp8b | WALS features predict training efficiency across 12 languages | Uses WALS as lambda parameterization |
| Wasserman Exp8b interleaved | Mixing languages degrades French grammar | French control checks for analogous interference |
| Wasserman Exp5 (Synthetic) | Synthetic languages with designed morphology | If this experiment succeeds, connects to synthetic language calibration |
| VM4AI Polytope topology | Rigid geometry enforces precise logical reasoning at inference time | Translates inference-time geometric constraint to training-time loss function |

## 8. Follow-on Experiments (planned, not pre-registered)

- **Exp2 (Sphere Loss)**: VM4AI's Sphere topology (Fluid/Creative) applied as representation norm constraint with higher lambda values. Edward Levin hypothesizes that sphere geometry may "mold morphology" by forcing angular clustering of related tokens.
- **Exp3 (potential)**: If Exp1 or Exp2 shows grammar improvement, test on Chinese (most morphologically impoverished language in exp8b, 75.8M tokens to 60% grammar)
- **Exp4 (potential)**: Combine geometric loss with synthetic language training data from Wasserman's Exp5

---

*Design finalized: 2026-04-02*
*Adam Wasserman + Edward Levin (VM4AI)*
