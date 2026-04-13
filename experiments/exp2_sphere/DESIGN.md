# Exp2: Sphere Loss

## Can Representation Geometry Synthesize Morphological Advantage?

## Pre-registration

**OSF registration**: [TODO: register before first training run]

## 1. Motivation

In VM4AI's cognitive topology framework, the Polytope and Sphere are fundamentally different geometric shapes serving different purposes. The Polytope (Rigid/Logic) enforces hard edges and precise constraints: "no wiggle room." The Sphere (Fluid/Creative) is smooth and round: "ideas slide and connect easily." At inference time, these topologies produce measurably different cognitive outputs across VM4AI's user base.

Exp1 translates the Polytope topology into a training-time loss (attention entropy minimization). Exp2 does the same for the Sphere topology: constraining the *representation space itself* to live on a hypersphere, forcing the model to organize knowledge through angular relationships rather than magnitude.

Edward Levin (VM4AI) hypothesizes that this may "mold morphology" by forcing grammatically related tokens into tight angular clusters, analogous to how French morphology naturally clusters inflected forms (mange/manges/mangent) through shared stems. He specifically recommends higher lambda values than Exp1, reflecting the Sphere's role as a more pervasive structural constraint.

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
| Edward's lambda recommendation | 1.50-1.85 | Higher (TBD with Edward) |

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

Concurrent experiments for the BabyLM 2026 submission ("Born Speaking French") provide new evidence that sharpens predictions for this experiment and, critically, transforms the Exp1/Exp2 interaction hypothesis (H2) from an open matrix into a directional prediction.

A 125M GPT-2 trained exclusively on French (92M words) was tested on English benchmarks using a vocabulary bridge (FR-EN word translations, no grammar or fine-tuning). Results reveal a **transfer gradient**:

| Transfer tier | Tasks | Bridge effect | Character |
|---|---|---|---|
| **Relational/logical** | RTE (+6.5pp), MRPC (+4.9pp) | Strong | Rigid, structural |
| **Semantic** | MNLI (+2.0pp), BoolQ (+1.8pp) | Weak | Mixed |
| **Discourse** | MultiRC (0), QQP (0), WSC (0) | None | Fluid, contextual |

The key observation for Exp2: discourse-level comprehension — the tier that does *not* transfer via a simple vocabulary bridge — requires richer contextual representation. This is precisely the domain where VM4AI's Sphere topology (fluid/creative, "ideas slide and connect easily") operates. If Polytope constrains the logical tier (Exp1), Sphere should constrain the associative/discourse tier.

This evidence allows us to replace the undirected Exp1/Exp2 interaction matrix with a specific directional prediction.

## 3. Hypotheses

### H0 (null)
Sphere Loss does not break the English grammar ceiling. Grammar accuracy remains <=50% regardless of lambda. This would suggest that representation geometry, like attention entropy, cannot substitute for morphological signal in the data.

### H1 (alternative)
Sphere Loss produces English grammar accuracy >50% sustained over 3+ consecutive checkpoints.

### H2 (directional interaction with Exp1, from BabyLM transfer gradient)
Sphere Loss and Polytope Loss affect **different dimensions of linguistic competence**, corresponding to different tiers of the BabyLM transfer gradient:

- **Polytope Loss** (Exp1) should preferentially improve relational/logical probes (agreement, binding, argument structure) — the tier that transfers cross-linguistically.
- **Sphere Loss** (Exp2) should preferentially improve **perplexity and discourse-level coherence** (next-token prediction in extended context, naturalness of generation) — the tier that requires richer contextual representation and does not transfer via a simple vocabulary bridge.

This replaces the original undirected 2x2 matrix with a falsifiable prediction: if both loss functions affect the same probes equally, the VM4AI topology distinction does not map onto the BabyLM transfer gradient, and the topologies are not targeting distinct aspects of linguistic structure.

**Basis**: The BabyLM dict-axioms experiment (Wasserman, 2026, unpublished) shows that relational concepts transfer cross-linguistically with rigid structure (vocabulary bridge), while discourse comprehension requires fluid contextual integration. VM4AI's Polytope (rigid) and Sphere (fluid) map onto exactly these two tiers.

### H3 (French control)
French grammar accuracy under Sphere Loss will not exceed its exp8b baseline (87%).

### H4 (lambda scaling)
Higher lambda values will be required compared to Exp1. Rationale: constraining the full representation space is a stronger intervention than constraining attention distributions, so the model needs more pressure to reorganize.

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

Sphere Loss lambda. Two parameterizations matching Exp1 structure:

**Lambda range**: TBD in consultation with Edward Levin. His recommendation is higher values than Exp1's 1.50-1.85 range. Preliminary range: 2.0-4.0 (to be finalized before pre-registration).

**Arm 1, BPE Fertility:**

| Run | Language | Lambda | Purpose |
|-----|----------|--------|---------|
| 1   | English  | TBD-low | Below threshold |
| 2   | English  | TBD-mid | Predicted sweet spot |
| 3   | English  | TBD-high | Above threshold |
| 7   | French   | TBD-mid | French control |

**Arm 2, WALS Composite:**

| Run | Language | Lambda | Purpose |
|-----|----------|--------|---------|
| 4   | English  | TBD-low | WALS low |
| 5   | English  | TBD-mid | WALS midpoint |
| 6   | English  | TBD-high | WALS high |
| 8   | French   | TBD-mid | French control |

### 4.3 Dependent variables

Measured every 1000 steps:

1. **Grammar probe accuracy** (primary outcome): same probes as exp8b and Exp1
2. **Validation perplexity**: held-out set from exp8b
3. **Mean representation norm**: to verify the loss is actually constraining norms
4. **Norm variance across layers**: to detect whether some layers resist the constraint
5. **Training loss decomposition**: CE component vs sphere component separately

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

Same 8-run structure as Exp1 with lambda values TBD.

**Total**: 8 runs x 100k steps = 800k steps
**Estimated compute**: ~2 days on 2x RTX 4090

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
- Hypothesis that higher lambda values can "mold morphology" through angular clustering
- Lambda range recommendation (TBD; Edward indicated higher values than Exp1's 1.50-1.85)
- VM4AI framework provides the theoretical grounding: if Sphere geometry shapes cognition at inference time, it should also shape learning at training time

### Adam Wasserman ([fractal-language](https://github.com/adamzwasserman/fractal-language))
- Cross-linguistic baselines from 12-language controlled ablation (exp8b)
- WALS morphological parameterization
- Experimental design, pre-registration framework, and analysis plan
- Training infrastructure and compute

### Intellectual property
- VM4AI geometric engine and cognitive topologies: Edward Levin & Karen Levin, CC-BY-NC-SA 4.0
- Morphological calibration methods and per-language lambda tuning: Subject to provisional patents held by Adam Wasserman
- Joint experimental results: Shared with attribution to both researchers

## 7. Open Questions for Edward

Before finalizing pre-registration:

1. What lambda range do you recommend for Sphere Loss? You mentioned "higher"; how much higher?
2. Should the norm constraint apply to all layer outputs, or only specific layers (early/late)?
3. Do you anticipate interaction effects if Polytope + Sphere are combined? (This would be Exp3.)
4. In VM4AI, the Sphere topology uses NMA (Native Meaning Alignment). Is there an analogue we should incorporate into the training-time translation?

## 8. Follow-on Experiments

- **Exp3 (potential)**: Combined Polytope + Sphere Loss, testing whether VM4AI's distinct topologies produce additive effects when applied simultaneously during training
- **Exp4 (potential)**: Apply the most effective geometric constraint to Chinese (most morphologically impoverished language in exp8b)

---

*Design drafted: 2026-04-02*
*Lambda values: TBD pending consultation with Edward Levin*
*Adam Wasserman + Edward Levin (VM4AI)*
