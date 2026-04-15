# Exp2: Sphere Loss

## Can Representation Geometry Synthesize Morphological Advantage?

## Pre-registration

**OSF registration**: [TODO: register before first training run]

## 1. Motivation

In VM4AI's cognitive topology framework, the Polytope and Sphere are fundamentally different geometric shapes serving different purposes. The Polytope (Rigid/Logic) enforces hard edges and precise constraints: "no wiggle room." The Sphere (Fluid/Creative) is smooth and round: "ideas slide and connect easily." At inference time, these topologies produce measurably different cognitive outputs across VM4AI's user base.

Exp1 translates the Polytope topology into a training-time loss (attention entropy minimization). Exp2 does the same for the Sphere topology: constraining the *representation space itself* to live on a hypersphere, forcing the model to organize knowledge through angular relationships rather than magnitude.

Edward Levin (VM4AI) originally proposed that Sphere Loss would likely require higher lambda values than Exp1, reflecting the intuition that constraining the representation manifold is broader than constraining attention distributions. We revise that assumption here.

The Sphere is not hypothesized to help by maximizing pressure. Its proposed role is to preserve and organize **high-entropy, abstract, interpretive structure** through angular relationships in representation space. Natural language, especially in its more abstract and less rigidly structured uses, is not low-entropy material. Because lambda pressure reduces entropy, an overly large Sphere penalty may over-compress the representational regime that Sphere geometry is meant to support.

The revised hypothesis is therefore not that stronger pressure is better, but that Sphere Loss should operate within a bounded calibration band: enough pressure to induce directional organization, but not enough to collapse representational flexibility. This shifts Exp2 from a fixed high-lambda assumption to a scheduled sweep beginning at **1.30** and increasing by **0.10** until performance reaches a cap or begins to show diminishing returns.

This is orthogonal to Exp1. Polytope Loss constrains the attention simplex; Sphere Loss constrains the representation manifold. They may affect different axes of the perplexity/accuracy space identified in Wasserman (2026).

## 2. Background

### Why a sphere?

Natural language semantics appear to be fundamentally directional. Word embedding research (word2vec, GloVe) showed that semantic relationships are captured by angular relationships between vectors, not by magnitudes. By constraining representations to a hypersphere:

1. **Angular clustering**: Grammatically related tokens must cluster by angle, not magnitude. Verb conjugations, agreement forms, and case-marked nouns would be forced into angular neighborhoods, similar to how morphological paradigms organize related forms.

2. **Elimination of magnitude shortcuts**: Without magnitude as an encoding dimension, the model cannot "hide" grammatical information in vector norms. All structure must be encoded directionally, which may force more explicit structural organization.

3. **Higher-dimensional structure**: Language is high-dimensional. A sphere constraint operates across all dimensions simultaneously, potentially capturing structural relationships that the simplex constraint of Polytope Loss (which operates per-attention-head) cannot.

### Relationship to Exp1

| Property | Exp1 (Polytope) | Exp2 (Sphere) |
|----------|----------------|---------------|
| Constrains | Attention distributions | Hidden representations |
| Geometry | Simplex (probabilities) | Hypersphere (unit norm) |
| Mechanism | Focus attention | Organize knowledge |
| Per-layer? | Yes (each attention head) | Yes (each layer output) |
| Edward's lambda recommendation | 1.50-1.85 | 1.30-start calibrated sweep (+0.10 increments) |

### Exp8b baselines (same tokenizer, same architecture)

| Language | Grammar (final) | Tokens to 60% | Val PPL | Tokens |
|----------|----------------|---------------|---------|--------|
| English  | 87%*           | 22.5M         | 74.4    | 780M   |
| French   | 87%            | 6.1M          | 37.7    | 780M   |

*Note: See Exp1 DESIGN.md for discussion of exp1 vs exp8b baseline discrepancy.

### The intervention

Add a sphere loss penalty that pushes hidden state representations toward unit norm:

```
sphere_penalty = mean((||h_l|| - 1)^2)  for each layer l
total_loss = CE(logits, labels) + lambda * sphere_penalty
```

Where ||h_l|| is the L2 norm of the hidden state output at layer l.

## 2.5 New evidence: BabyLM cross-linguistic transfer (Wasserman, 2026, unpublished)

Concurrent experiments for the BabyLM 2026 submission ("Right Tool, Right Job") provide new evidence that sharpens predictions for this experiment and, critically, transforms the Exp1/Exp2 interaction hypothesis (H2) from an open matrix into a directional prediction.

A 125M GPT-2 trained exclusively on French (92M words) was evaluated on English GLUE tasks under a controlled experiment grid isolating four cross-lingual adaptation levers (English-native LoRA baseline; inference-time vocabulary axioms; tuned-rank LoRA; LoRA fine-tuning on French-translated task data). The fine-tuning lever (rank-16 LoRA on French-translated task data) reveals a stable **transfer gradient** across GLUE tasks:

| Transfer tier | Tasks | E3 effect (French-translated LoRA, epoch 3) | Character |
|---|---|---|---|
| **Relational/logical** | RTE ($+$7.91pp), MRPC ($+$2.45pp) | Strong | Rigid, structural |
| **Semantic** | MNLI ($+$4.93pp), BoolQ ($+$3.67pp) | Moderate | Mixed |
| **Discourse** | MultiRC (0), QQP (0), WSC ($+$1.93pp) | Weak or none | Fluid, contextual |

The key observation for Exp2: discourse-level comprehension — the tier that does *not* transfer even under fine-tuning with translated task data — requires richer contextual representation. This is precisely the domain where VM4AI's Sphere topology (fluid/creative, "ideas slide and connect easily") operates. If Polytope constrains the logical tier (Exp1), Sphere should constrain the associative/discourse tier.

This evidence allows us to replace the undirected Exp1/Exp2 interaction matrix with a specific directional prediction. (Earlier drafts of this section cited an earlier BabyLM "dict-axioms" vocabulary-bridge experiment as the source of the transfer-gradient numbers; that result collapsed under placebo control, see §4.3 retraction note. The transfer gradient as stated above is from the fine-tuning lever, which is robust to the placebo and tokenizer-swap confounds documented in BabyLM §6.2 and §6.4.)

## 3. Hypotheses

### H0 (null)
Sphere Loss does not break the English grammar ceiling. Grammar accuracy remains <=50% regardless of lambda. This would suggest that representation geometry, like attention entropy, cannot substitute for morphological signal in the data.

### H1 (alternative)
Sphere Loss produces English grammar accuracy >50% sustained over 3+ consecutive checkpoints.

### H2 (directional interaction with Exp1, from BabyLM fine-tuning transfer gradient)
Sphere Loss and Polytope Loss affect **different dimensions of linguistic competence**, corresponding to different tiers of the BabyLM fine-tuning transfer gradient:

- **Polytope Loss** (Exp1) should preferentially improve relational/logical probes (agreement, binding, argument structure), the tier that transfers strongly under French-translated-task LoRA.
- **Sphere Loss** (Exp2) should preferentially improve **perplexity and discourse-level coherence** (next-token prediction in extended context, naturalness of generation), the tier that requires richer contextual representation and does not transfer under fine-tuning of translated task data.

This replaces the original undirected 2x2 matrix with a falsifiable prediction: if both loss functions affect the same probes equally, the VM4AI topology distinction does not map onto the BabyLM transfer gradient, and the topologies are not targeting distinct aspects of linguistic structure.

**Basis**: The BabyLM fine-tuning transfer gradient (Wasserman, 2026, unpublished; Figure 1 of the BabyLM submission) shows that relational concepts (RTE, MRPC) transfer cross-linguistically with rigid structural alignment ($+$2.45 to $+$7.91pp), while discourse-level comprehension (MultiRC, QQP, WSC) requires fluid contextual integration and does not meaningfully transfer (0 to $+$1.93pp). VM4AI's Polytope (rigid) and Sphere (fluid) map onto exactly these two tiers. (Earlier drafts grounded this basis in the BabyLM dict-axioms experiment, which collapsed under placebo control; the prediction survives because the fine-tuning gradient is independent evidence for the same stratification.)

### H3 (French control)
French grammar accuracy under Sphere Loss will not exceed its exp8b baseline (87%).

### H4 (lambda calibration)
Sphere Loss will show a bounded effective pressure range rather than a monotonic benefit from larger lambda values. Starting from a low lambda, performance should improve up to a moderate band, then flatten or degrade once additional pressure begins to suppress the high-entropy representational structure Sphere geometry is intended to organize.

## 4. Experimental Design

### 4.1 Controlled variables (identical to Exp1 and exp8b)

- **Architecture**: 125M GPT-2 style (12 layers, d_model=768, 12 heads, d_ff=3072)
- **Tokenizer**: Exp8b joint BPE (50k vocab). MUST NOT be retrained.
- **Optimizer**: AdamW, lr=6e-4, weight_decay=0.01
- **Random seed**: 42
- **Sequence length**: 512
- **Batch size**: 16
- **Data**: Same chunks from exp8b (English, French)

### 4.2 Independent variable

Sphere Loss lambda. Two parameterizations matching Exp1 structure.

**Lambda schedule:** Rather than assuming that Sphere Loss requires a fixed high-pressure regime, Exp2 uses a calibrated upward sweep. The schedule begins at **1.30** and increases by **0.10** until performance reaches a cap or exhibits diminishing returns. This reflects the revised theoretical claim that Sphere geometry should organize high-entropy representation space without over-compressing it.

**Arm 1, BPE Fertility:**

| Run | Language | Lambda | Purpose |
|-----|----------|--------|---------|
| 1   | English  | 1.30 | Low-start calibration |
| 2   | English  | 1.40 | Upward sweep |
| 3   | English  | 1.50 | Upward sweep |
| 4   | English  | 1.60 | Upward sweep |
| 5   | English  | 1.70 | Upward sweep |
| 6   | English  | 1.80 | Upward sweep |
| 7   | French   | 1.40 | French control |
| 8   | French   | 1.50 | French control |

This initial matrix can be trimmed or extended depending on compute budget and early saturation patterns. If a stronger contrast between fertility and WALS parameterization is preferred, the same sweep logic can be retained while assigning different mid-band values by language.

**Arm 2, WALS Composite:**

| Run | Language | Lambda | Purpose |
|-----|----------|--------|---------|
| 9   | English  | 1.30 | WALS low-start calibration |
| 10  | English  | 1.40 | WALS upward sweep |
| 11  | English  | 1.50 | WALS upward sweep |
| 12  | English  | 1.60 | WALS upward sweep |
| 13  | English  | 1.70 | WALS upward sweep |
| 14  | English  | 1.80 | WALS upward sweep |
| 15  | French   | 1.40 | French control |
| 16  | French   | 1.50 | French control |

### 4.3 Dependent variables

Measured every 1000 steps (except where noted):

1. **Grammar probe accuracy** (primary outcome): same probes as exp8b and Exp1
2. **Validation perplexity**: held-out set from exp8b
3. **Mean representation norm**: to verify the loss is actually constraining norms
4. **Norm variance across layers**: to detect whether some layers resist the constraint
5. **Training loss decomposition**: CE component vs sphere component separately
6. **BLI Procrustes alignment to exp8b French** (structural outcome; measured at 10k, 25k, 50k, 100k steps): orthogonal Procrustes fit and word-translation p@1 between each Sphere-regularized English checkpoint's embedding matrix and the exp8b final French checkpoint's embedding matrix, using a seed dictionary of $\sim$200 high-frequency EN-FR concept pairs. **Pre-registered prediction**: Sphere Loss operates on representation norms rather than attention distributions; if it induces the angular-organization structure its theoretical motivation posits, BLI alignment to French should be at least as high as, and possibly higher than, the unregularized English baseline — because angular-organized representations are exactly what orthogonal Procrustes can most easily align. If Sphere Loss lowers BLI alignment to French relative to baseline, it has organized English representations into a geometry incompatible with the French geometry, which would be an informative negative result (Sphere Loss is changing representations but in a language-specific direction, not toward a shared substrate).
7. **Tokenizer-swap sanity check** (run once at end per lambda, not every 1000 steps): per BabyLM §6.4's finding that single-token log-probability scoring at child scale moves by $\sim$7.7pp on GLUE-axiomatic from tokenizer alone, any grammar-probe improvement claimed for Sphere Loss should be verified with a tokenizer swap. Instability under tokenizer swap indicates a lexical-distributional artifact rather than a representation-level effect.

**Retracted prior citation.** Earlier drafts of this document referenced the BabyLM "dict-axioms" experiment as evidence for cross-lingual conceptual transfer of discourse-level material. That result collapsed under placebo control (BabyLM §6.2) and is retained in the BabyLM paper only as a methodological negative result. The transfer gradient this experiment's H1/H2 are built on is now grounded in the BabyLM fine-tuning lever (Figure 1 of the submission) and the BLI triangulation (§5.1), both of which survive all placebo and tokenizer-swap confounds documented in the BabyLM paper.

### 4.4 Run length

100,000 steps (~200M tokens), matching Exp1.

### 4.5 Success criteria

Same as Exp1:

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| Grammar breakthrough | >50% for 3+ consecutive checkpoints | Exceeds chance + noise |
| Perplexity improvement | <74.4 (exp8b English baseline) | Faster convergence |
| French interference | French grammar <80% | Sphere Loss harms natural morphological signal |

### 4.6 Run matrix

| Run | Language | Arm | Lambda | Steps | Purpose |
|-----|----------|-----|--------|-------|---------|
| 1   | English  | Fertility | 1.30 | 100k | Low-start calibration |
| 2   | English  | Fertility | 1.40 | 100k | Upward sweep |
| 3   | English  | Fertility | 1.50 | 100k | Upward sweep |
| 4   | English  | Fertility | 1.60 | 100k | Upward sweep |
| 5   | English  | Fertility | 1.70 | 100k | Upward sweep |
| 6   | English  | Fertility | 1.80 | 100k | Upward sweep |
| 7   | French   | Fertility | 1.40 | 100k | French control |
| 8   | French   | Fertility | 1.50 | 100k | French control |
| 9   | English  | WALS | 1.30 | 100k | WALS low-start calibration |
| 10  | English  | WALS | 1.40 | 100k | WALS upward sweep |
| 11  | English  | WALS | 1.50 | 100k | WALS upward sweep |
| 12  | English  | WALS | 1.60 | 100k | WALS upward sweep |
| 13  | English  | WALS | 1.70 | 100k | WALS upward sweep |
| 14  | English  | WALS | 1.80 | 100k | WALS upward sweep |
| 15  | French   | WALS | 1.40 | 100k | French control |
| 16  | French   | WALS | 1.50 | 100k | French control |

**Total**: 16 runs x 100k steps = 1.6M steps
**Estimated compute**: ~4 days on 2x RTX 4090

## 5. Analysis Plan

### Primary analysis
Same binary outcome as Exp1: did any run break the 50% grammar threshold?

### Cross-experiment analysis (Exp1 + Exp2)
The key analysis is comparing results across both experiments:

| Exp1 result | Exp2 result | Interpretation |
|-------------|-------------|----------------|
| No grammar improvement | No grammar improvement | Strongest support for Language-Only Hypothesis |
| PPL improvement only | Grammar improvement | Different geometric axes target different learning dimensions |
| Grammar improvement | No grammar improvement | Attention focus is the key mechanism |
| Grammar improvement | Grammar improvement | Morphological advantage is synthesizable through multiple geometric paths |

### Reporting commitment
All results reported regardless of outcome. The cross-experiment comparison is the primary contribution even if all individual results are null.

## 6. Collaboration

### Edward Levin ([VM4AI](https://vm4ai.com))
- Sphere Loss concept, derived from VM4AI's Sphere cognitive topology (Fluid/Creative: "a smooth, round shape; ideas slide and connect easily")
- Revised hypothesis that Sphere Loss should operate within a bounded calibration band rather than a fixed high-pressure regime
- Proposed low-start lambda schedule beginning at 1.30 with +0.10 increments until performance flattens or degrades
- VM4AI framework provides the theoretical grounding: if Sphere geometry shapes cognition at inference time, it should also shape learning at training time.

### Adam Wasserman ([fractal-language](https://github.com/adamzwasserman/fractal-language))
- Cross-linguistic baselines from 12-language controlled ablation (exp8b)
- WALS morphological parameterization
- Experimental design, pre-registration framework, and analysis plan
- Training infrastructure and compute

### Intellectual property
- VM4AI geometric engine and cognitive topologies: Edward Levin & Karen Levin, CC-BY-NC-SA 4.0
- Joint experimental results: Shared with attribution to both researchers

## 7. Open Questions for Edward

Before finalizing pre-registration:

1. Should the 1.30-start, +0.10 lambda sweep stop at a fixed ceiling, or should it terminate empirically once validation perplexity and probe gains flatten?
2. Should the norm constraint apply to all layer outputs, or only specific layers (early vs late)?
3. Do you anticipate interaction effects if Polytope + Sphere are combined? (This would become a later experiment.)
4. In VM4AI, the Sphere topology uses NMA (Native Meaning Alignment). Is there a training-time analogue worth incorporating here?

## 8. Follow-on Experiments

- **Exp3 (potential)**: Combined Polytope + Sphere Loss, testing whether VM4AI's distinct topologies produce additive effects when applied simultaneously during training
- **Exp4 (potential)**: Apply the most effective geometric constraint to Chinese (most morphologically impoverished language in exp8b)

---

*Design revised: 2026-04-15*
*Lambda schedule updated to low-start calibrated sweep*
*Adam Wasserman + Edward Levin (VM4AI)*
