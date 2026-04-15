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

## 2.5 New evidence: BabyLM cross-linguistic transfer and BLI triangulation (Wasserman, 2026, unpublished)

Concurrent experiments for the BabyLM 2026 submission ("Right Tool, Right Job") provide new evidence that sharpens predictions for this experiment.

**BLI triangulation (new 2026-04-14).** Direct measurement of cross-lingual embedding alignment via orthogonal Procrustes: the French BabyLM model aligns to GPT-2's embedding space at word-translation p@1 of 66.7% (32× chance), but to a matched-architecture 125M English model from exp8b that failed to acquire English grammar at only 25.0%. The competence-gating pattern in embedding geometry is the cleanest pre-existing evidence that training dynamics and representation geometry are coupled — a Polytope intervention that changes training dynamics should, if it changes *what the model learns to represent* rather than merely perplexity, also change the embedding geometry in a BLI-measurable way. Exp 1 adopts BLI alignment to exp8b French as an additional outcome metric (see §4.3).

**The cross-task transfer gradient (fine-tuning, not dict-axioms).** BabyLM Figure 1 documents a stable gradient across GLUE tasks under the fine-tuning lever:

| Transfer tier | Tasks | E3 effect (French-translated LoRA, epoch 3) | Character |
|---|---|---|---|
| **Relational/logical** | RTE (+7.91pp), MRPC (+2.45pp) | Strong | Rigid, structural |
| **Semantic** | MNLI (+4.93pp), BoolQ (+3.67pp) | Moderate | Mixed |
| **Discourse** | MultiRC (0), QQP (0), WSC (+1.93pp) | Weak or none | Fluid, contextual |

Relational concepts (entailment, paraphrase) transfer cross-linguistically; discourse-level comprehension does not. This gradient maps onto VM4AI's topology distinction: Polytope (rigid/logic) corresponds to the tier that transfers; Sphere (fluid/creative) corresponds to the tier that does not. This evidence motivates H4 (not in the original design).

**Retracted citation.** Earlier drafts of this document cited a BabyLM "dict-axioms" experiment as evidence for the transfer gradient; that specific result collapsed under placebo control (BabyLM §6.2) and is retained in the BabyLM paper only as a methodological negative result. The cross-task transfer gradient as stated here is from the fine-tuning lever, which is not subject to the placebo-control issue and is robust.

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

**Basis**: The BabyLM fine-tuning transfer gradient (Wasserman, 2026, unpublished; §2.5 above and Figure 1 of the BabyLM submission) documents that relational/logical task structures (RTE, MRPC) transfer across languages under light LoRA adaptation with effect sizes of $+$2.45 to $+$7.91pp, while discourse-level structures (MultiRC, QQP, WSC) transfer weakly or not at all. The gradient is observed in the fine-tuning lever, which is robust to the placebo and tokenizer-swap confounds documented in BabyLM §6.2 and §6.4. If the Polytope topology corresponds to rigid/logical structure (as VM4AI posits), its training-time analogue should preferentially improve the same tier of linguistic competence that transfers cross-linguistically under fine-tuning. This prediction is falsifiable: if Polytope Loss improves all probe types uniformly, the VM4AI topology-to-transfer mapping does not hold. (Earlier drafts grounded this basis in the BabyLM dict-axioms experiment, which collapsed under placebo control; the prediction survives because the fine-tuning gradient is independent evidence for the same stratification.)

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

Measured every 1000 steps (except where noted):

1. **Grammar probe accuracy** (primary outcome): same probes as exp8b
2. **Validation perplexity**: held-out set from exp8b
3. **Mean attention entropy**: to verify the loss is actually affecting attention patterns
4. **Training loss decomposition**: CE component vs polytope component separately
5. **BLI Procrustes alignment to exp8b French** (structural outcome; measured at 10k, 25k, 50k, 100k steps): orthogonal Procrustes fit and word-translation p@1 between each checkpoint's embedding matrix and the exp8b final French checkpoint's embedding matrix, using a seed dictionary of $\sim$200 high-frequency English-French concept pairs. **Pre-registered prediction**: if Polytope Loss genuinely changes what the English model represents (not only how fast it converges on perplexity), BLI alignment to French should increase monotonically with training under Polytope-regularized runs, and the final-checkpoint alignment should exceed the baseline (un-regularized) English-to-French alignment at 100k steps. If grammar-probe accuracy improves (H1) but BLI alignment to French does not, the intervention is changing task-competence proxies without changing representation geometry — a null result for the Polytope-replicates-morphology hypothesis at the representation level. Adopting BLI as a secondary outcome metric is motivated by the BabyLM triangulation pilot (§2.5) and makes the experiment informative even if grammar probes are null.
6. **Tokenizer-swap sanity check** (run once at end, not every 1000 steps): re-evaluate the final English-Polytope checkpoint with the exp8b English tokenizer swapped for the exp8b French tokenizer. Based on the BabyLM §6.4 finding that single-token log-probability scoring at child scale moves by $\sim$7.7pp on GLUE-axiomatic from tokenizer alone, any grammar-probe improvement claimed for Polytope Loss should be stable to within a few percentage points under tokenizer swap. If the improvement disappears or reverses under tokenizer swap, the result is a lexical-distributional artifact rather than a representation-level change.

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
