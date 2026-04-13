# Wittgenstein or Plato? Geometric Constraints and Language-Independent Representation in Small Language Models

## A Pre-Registration of Three Experiments

**Authors:** Adam Wasserman & Edward Levin (VM4AI)

**Date:** 2026-04-13

**OSF Registration:** [TODO: file before first training run or analysis]

---

## Abstract

We pre-register three experiments testing whether geometric constraints applied during language model training can replicate the structural advantages that morphologically rich languages provide naturally, and whether the resulting representations are language-independent.

Two independent lines of research motivate this work. Wasserman (2026) established that morphologically rich languages train dramatically more efficiently than morphologically poor languages on identical architectures, with WALS morphological features predicting training efficiency at r=-0.88. Levin & Levin (2025) developed VM4AI, a geometry-aware cognitive controller demonstrating that topological constraints shape AI reasoning at inference time. Concurrent unpublished results from BabyLM 2026 experiments (Wasserman, 2026) reveal that a French-trained model possesses language-independent conceptual representations accessible via a simple vocabulary bridge, with a transfer gradient — relational concepts transfer strongly, discourse comprehension does not — that maps directly onto VM4AI's topology distinction between Polytope (rigid/logic) and Sphere (fluid/creative).

Together, these findings suggest a synthesis: meaning is acquired through the structure of language (Wittgenstein) but represented as language-independent form (Plato). We test this synthesis through three experiments: (1) Polytope Loss, attention entropy minimization during training; (2) Sphere Loss, representation norm constraints during training; (3) Vector Invariance, cross-linguistic convergence of latent representations. We pre-register directional predictions for each experiment derived from the BabyLM transfer gradient, and specify the cross-experiment analysis plan that connects all three.

This work is framed as a contribution to both machine learning and linguistics. We propose that language model training dynamics function as telemetry, measurement instruments that reveal properties already present in language, rather than generating novel linguistic behaviour. The experimental programme tests whether this telemetry reads out a universal cognitive substrate (Plato) or language-specific structure (Wittgenstein), with implications for how linguists might use language models as fieldwork instruments.

---

## 1. Introduction

### 1.1 The question

Is meaning in the structure of language, or in the forms that language points to?

This is not a new question. Wittgenstein (1953) argued that meaning is use: that what a word means is constituted by how it functions in language. Plato argued that meaning exists independently of any particular expression: that language points to forms that transcend it. The debate has continued for millennia because it has been, until now, empirically intractable. We cannot examine the meaning of a word independently of the language that expresses it.

Language models offer a new kind of evidence. A model trained exclusively on French, evaluated on English tasks it has never seen, either succeeds or fails. If it succeeds — if concepts learned through French structure are accessible in English — then something language-independent exists in the model's internal representations. If it fails, meaning may be more deeply bound to its linguistic medium than the Platonic view suggests.

We have both kinds of evidence, and the pattern they form is the basis for the experiments we pre-register here.

### 1.2 LLMs as telemetry

The framing of this work departs from the standard machine learning orientation. We are not trying to make models better. We are using models as measurement instruments to reveal properties of language.

This claim rests on a specific observation: when identical 125M-parameter transformers are trained on matched corpora across 12 languages with every variable held constant except the training language, the resulting training dynamics — perplexity curves, grammar probe trajectories, tokens-to-threshold — correlate with typological properties catalogued independently by linguists in the World Atlas of Language Structures (WALS). The correlation is r=-0.88 for WALS VerbSynth and r=-0.78 for WALS Agreement (Wasserman, 2026). The model is not discovering these properties. It is reading them out, measuring the same thing that typological linguists measure through fieldwork, but through a different instrument.

If this framing is correct, then the experiments pre-registered here are not primarily about model architecture or training techniques. They are about what the telemetry reveals: whether the cognitive structure that morphologically rich languages make visible exists independently of any particular language (Plato), or is constituted by the language itself (Wittgenstein).

### 1.3 Two independent research programmes

**Wasserman (2026)** conducted controlled cross-linguistic ablation experiments (exp8b) across 12 languages — English, French, Spanish, Finnish, Russian, Vietnamese, Chinese, and four synthetic languages — using identical 125M GPT-2 architectures, the same joint BPE tokenizer, the same hyperparameters, and the same training procedure. The central finding: morphological structure, not data quantity, determines training efficiency. French reaches grammatical competence in 6.1M tokens; English requires 22.5M tokens to reach the same threshold. At matched token counts, French achieves validation perplexity of 37.7 versus English's 74.4. WALS morphological features (VerbSynth, Agreement, TAM, Fusion) predict these differences with high fidelity.

**Levin & Levin (2025)** developed VM4AI (Virtual Machine for AI), a geometry-aware cognitive controller that enforces topological constraints on AI reasoning. VM4AI defines cognitive topologies — Polytope (rigid, hard-edged, logical), Sphere (smooth, fluid, creative), and others — that measurably shape cognitive output at inference time. The key insight for this collaboration: if geometric structure shapes cognition at inference time, it may also shape learning at training time. Levin independently proposed that latent vectors for concepts may be model-invariant: that a concept occupies the same coordinate in latent space regardless of which model encodes it, because the vector represents a property of the concept, not a property of the model.

These programmes converge on the same question from different directions. Wasserman's data shows that linguistic structure determines how efficiently a model learns. Levin's framework proposes that geometric structure shapes what a model learns. Together, they ask: can geometric constraints replicate what linguistic structure provides naturally, and is the destination the same regardless of the path?

### 1.4 New evidence from BabyLM (unpublished)

Concurrent experiments for the BabyLM 2026 submission ("Born Speaking French"; Wasserman, 2026, unpublished) provide critical evidence that sharpens the predictions pre-registered here.

A 125M-parameter GPT-2 trained exclusively on 92M words of French was evaluated on English benchmarks it had never seen, under three conditions: bare English input, a "dict-axioms" vocabulary bridge (FR-EN word translations prepended to the prompt, with no grammar, syntax, or fine-tuning), and full French translation. The dict-axioms condition is the critical test: it provides only lexical mapping, isolating whether the model possesses language-independent conceptual representations.

**Results:**

| Task | Bare English | Dict-axioms | LoRA fine-tuned | Bridge effect |
|------|-------------|-------------|-----------------|---------------|
| RTE (entailment) | 47.5% | **54.0%** | 53.24% | +6.5pp |
| MRPC (paraphrase) | 31.9% | **36.8%** | — | +4.9pp |
| MNLI (3-way entailment) | 32.2% | 34.2% | — | +2.0pp |
| BoolQ (comprehension) | 57.0% | 58.8% | — | +1.8pp |
| MultiRC (reading) | 53.8% | 53.8% | — | 0 |
| QQP (duplicate detection) | 62.6% | 62.6% | — | 0 |
| WSC (coreference) | 61.5% | 61.5% | — | 0 |

Two findings are load-bearing for this pre-registration:

**First**, the dict-axioms bridge enables a French model to perform English entailment reasoning better than gradient-based fine-tuning on English data. The model "knows" what entailment is. It learned the concept from French text alone. The concept exists in the model's internal representations independent of surface language. This is direct evidence for language-independent representation, and a pilot result for Experiment 3 (Vector Invariance).

**Second**, the transfer is not uniform. A gradient emerges:

| Transfer tier | Character | Tasks | Bridge effect |
|---|---|---|---|
| **Relational/logical** | Rigid, structural | RTE, MRPC | Strong (+4.9 to +6.5pp) |
| **Semantic** | Mixed | MNLI, BoolQ | Weak (+1.8 to +2.0pp) |
| **Discourse** | Fluid, contextual | MultiRC, QQP, WSC | None (0pp) |

This gradient maps directly onto VM4AI's topology distinction. Polytope (rigid/logic) corresponds to the tier that transfers; Sphere (fluid/creative) corresponds to the tier that does not. This mapping transforms our interaction hypothesis from an undirected possibility matrix into a directional prediction, and motivates new hypotheses in all three experiments.

### 1.5 The Wittgenstein-Plato synthesis

The BabyLM evidence, combined with the exp8b cross-linguistic results, suggests a synthesis:

**Wittgenstein is right about acquisition.** How a language structures its morphology determines how efficiently a model learns. French, with its gender agreement, verb conjugation, and morphological composition, creates a denser learning signal per token. Meaning is use, and French encodes more meaning in its patterns of use. The evidence: French at 92M words matches or exceeds what English achieves at 3B+ words on grammatical competence benchmarks.

**Plato is right about representation.** Once a concept is acquired, it exists as a language-independent form. Entailment learned through French is the same entailment tested in English. The concept is not constituted by French grammar; French grammar was merely the instrument that made it visible. The evidence: a vocabulary bridge (no grammar, no syntax, no training) is sufficient to access French-learned entailment for English tasks, outperforming gradient-based fine-tuning.

| | Acquisition (learning) | Representation (knowledge) |
|---|---|---|
| **Who is right** | Wittgenstein | Plato |
| **What matters** | Structure of language | Forms transcending language |
| **Evidence** | French 20x more efficient than English | Entailment transfers with word bridge only |
| **Mechanism** | Morphological redundancy = denser signal | Shared latent geometry across languages |

The three experiments pre-registered here test this synthesis. Experiments 1 and 2 test the Wittgensteinian side: can geometric constraints replicate what morphological structure provides during acquisition? Experiment 3 tests the Platonic side: does the destination — the latent representation of concepts — converge across languages?

---

## 2. General Methods

### 2.1 Architecture and baselines

All training experiments use the same architecture and configuration as exp8b (Wasserman, 2026), ensuring direct comparability:

- **Architecture:** 125M GPT-2 (12 layers, d_model=768, 12 attention heads, d_ff=3072)
- **Tokenizer:** Exp8b joint BPE (50k vocabulary). The tokenizer is frozen and must not be retrained.
- **Optimizer:** AdamW, lr=6e-4, weight_decay=0.01
- **Random seed:** 42
- **Sequence length:** 512
- **Batch size:** 16
- **Training data:** Same chunked corpora from exp8b (English, French)

### 2.2 Baselines from exp8b

| Language | Grammar (final) | Tokens to 60% grammar | Val PPL | Total tokens |
|----------|----------------|----------------------|---------|-------------|
| English | 87%* | 22.5M | 74.4 | 780M |
| French | 87% | 6.1M | 37.7 | 780M |

*English reached 87% grammar only in the balanced multilingual exp8b configuration (batch=16). In the English-only exp1 (batch=2), grammar was locked at 40%. Exp8b results are used as baselines throughout since all experiments here use the same batch configuration.

### 2.3 Evaluation

**Grammar probes** (primary outcome): Minimal pair forced-choice probes testing subject-verb agreement, gender agreement, article selection, and collocational knowledge. Evaluated every 1,000 training steps.

**Validation perplexity:** Held-out set from exp8b, computed every 1,000 steps.

**BabyLM suite** (for cross-referencing): BLiMP, BLiMP supplement, GLUE (7 tasks), EWoK, Entity Tracking (the same benchmarks used in the BabyLM experiments that motivate the sharpened predictions).

### 2.4 Reporting commitment

All results will be reported regardless of outcome. Null results are scientifically informative: they constrain the space of possible explanations and strengthen the Language-Only Hypothesis by showing that the grammar deficit cannot be overcome through training dynamics alone. The cross-experiment comparison (Section 7) is the primary contribution even if all individual results are null.

---

## 3. Experiment 1: Polytope Loss

### 3.1 Rationale

The Polytope topology in VM4AI enforces rigid, hard-edged logical reasoning: "a shape with hard edges, no wiggle room" (Levin & Levin, 2025). At inference time, it produces precise, structured cognitive output. Translated to training time, the Polytope becomes an attention entropy minimization penalty: forcing the model to attend to specific tokens rather than distributing attention broadly.

The hypothesis connecting this to cross-linguistic training dynamics: if morphological redundancy works by constraining attention to focus on structurally relevant tokens (agreement markers, conjugation suffixes, gender concordance), then explicit attention entropy minimization may replicate this constraint for languages whose morphology provides no such signal.

### 3.2 Intervention

Add a weighted attention entropy penalty to the standard cross-entropy loss:

```
total_loss = CE(logits, labels) + λ × mean(H(attention_weights))
```

Where H is the Shannon entropy of the attention distribution across all heads and layers.

### 3.3 Lambda parameterization

Two independent parameterizations of λ, enabling comparison of theoretically motivated scaling:

**Arm 1, BPE Fertility (Levin):** Lambda scaled by the ratio of BPE tokens to whitespace-delimited words. Languages that expand more under tokenization receive higher lambda, correcting for tokenization fragmentation. Values based on Levin's observed sweet-spot range: 1.50, 1.65, 1.85.

**Arm 2, WALS Composite (Wasserman):** Lambda scaled by morphological features from WALS (22A VerbSynth + 29A Agreement + 21B TAM + 20A Fusion). Higher lambda for lower WALS scores (languages with less morphology need more regularization). Normalized to the same [1.50, 1.85] range for direct comparison with Arm 1.

### 3.4 Run matrix

| Run | Language | Arm | Lambda | Purpose |
|-----|----------|-----|--------|---------|
| 1 | English | Fertility | 1.50 | Below sweet spot |
| 2 | English | Fertility | 1.65 | Sweet spot midpoint |
| 3 | English | Fertility | 1.85 | Above sweet spot |
| 4 | English | WALS | 1.50 | WALS low |
| 5 | English | WALS | 1.73 | WALS midpoint |
| 6 | English | WALS | 1.85 | WALS high |
| 7 | French | Fertility | 1.65 | French control |
| 8 | French | WALS | 1.50 | French control |

**Total:** 8 runs × 100,000 steps = 800k steps (~2 days on 2× RTX 4090)

### 3.5 Dependent variables (measured every 1,000 steps)

1. Grammar probe accuracy (primary outcome)
2. Validation perplexity
3. Mean attention entropy (verification that the loss is affecting attention patterns)
4. Training loss decomposition (CE component vs. Polytope component)

### 3.6 Hypotheses

**H0 (null, from orthogonality finding):** Neither parameterization breaks the grammar ceiling. English grammar accuracy under Polytope Loss will remain within measurement noise of the exp8b baseline (≤50% on grammar probes) at all lambda values tested. Perplexity may improve.

*Basis:* Exp1 showed English PPL improved from 1340 to 777 (42% reduction) while grammar remained locked at 40%. If the grammar deficit is structural (absent from the data), no loss function can recover it.

**H1 (alternative):** At least one parameterization produces English grammar accuracy >50% sustained over 3 or more consecutive checkpoints (3,000+ steps).

**H2 (conditional on H1):** If grammar improvement occurs, WALS-derived lambda produces equal or greater improvement than fertility-derived lambda, since WALS features are stronger predictors of training efficiency (r=-0.88) than tokenization-based proxies.

**H3 (French control):** French grammar accuracy under Polytope Loss will not exceed its exp8b baseline (87%). Morphologically rich languages already provide the regularization the loss term attempts to simulate; adding it is redundant at best and harmful at worst.

**H4 (probe stratification, from BabyLM transfer gradient):** If Polytope Loss improves grammar (H1 supported), the improvement will be non-uniform across probe types. Probes testing relational/logical structure (agreement, binding, argument structure) will improve before and more than probes testing discourse-level or pragmatic phenomena.

*Basis:* The BabyLM dict-axioms experiment (Wasserman, 2026, unpublished) demonstrates that relational/logical competence transfers cross-linguistically with only a vocabulary bridge, while discourse-level comprehension does not. If the Polytope topology corresponds to rigid/logical structure (as VM4AI posits), its training-time analogue should preferentially improve the same tier. This prediction is falsifiable: if Polytope Loss improves all probe types uniformly, the topology-to-transfer mapping does not hold.

### 3.7 Success criteria

| Criterion | Threshold | Meaning |
|-----------|-----------|---------|
| Grammar breakthrough | >50% for 3+ consecutive checkpoints | Exceeds chance + noise |
| Perplexity improvement | <74.4 (exp8b English baseline) | Faster convergence |
| French interference | French grammar <80% | Polytope Loss harms natural morphological signal |

---

## 4. Experiment 2: Sphere Loss

### 4.1 Rationale

The Sphere topology in VM4AI is smooth and fluid: "a smooth, round shape; ideas slide and connect easily" (Levin & Levin, 2025). Where the Polytope constrains the attention simplex (where the model looks), the Sphere constrains the representation manifold (how the model organizes knowledge). Translated to training time, the Sphere becomes a representation norm constraint: forcing hidden states toward unit norm, so all structure must be encoded directionally.

Levin hypothesizes that this may "mold morphology" by forcing grammatically related tokens into tight angular clusters, analogous to how French morphology naturally clusters inflected forms (mange/manges/mangent) through shared stems.

### 4.2 Intervention

Add a sphere loss penalty that pushes hidden state representations toward unit norm:

```
sphere_penalty = mean((||h_l|| - 1)²)  for each layer l
total_loss = CE(logits, labels) + λ × sphere_penalty
```

Where ||h_l|| is the L2 norm of the hidden state output at layer l.

### 4.3 Lambda parameterization

Same two-arm structure as Experiment 1 (BPE Fertility and WALS Composite), with a higher lambda range. Levin recommends higher values than Exp1's 1.50–1.85, reflecting the Sphere's role as a more pervasive structural constraint. Preliminary range: 2.0–4.0, to be finalized with Levin before registration.

### 4.4 Run matrix

Same 8-run structure as Experiment 1 with lambda values TBD.

**Total:** 8 runs × 100,000 steps = 800k steps (~2 days on 2× RTX 4090)

### 4.5 Dependent variables (measured every 1,000 steps)

1. Grammar probe accuracy (primary outcome)
2. Validation perplexity
3. Mean representation norm (verification that the loss is constraining norms)
4. Norm variance across layers (whether some layers resist the constraint)
5. Training loss decomposition (CE component vs. Sphere component)

### 4.6 Hypotheses

**H0 (null):** Sphere Loss does not break the English grammar ceiling. Grammar accuracy remains ≤50% regardless of lambda.

**H1 (alternative):** Sphere Loss produces English grammar accuracy >50% sustained over 3+ consecutive checkpoints.

**H2 (directional interaction with Exp1, from BabyLM transfer gradient):** Sphere Loss and Polytope Loss affect different dimensions of linguistic competence, corresponding to different tiers of the BabyLM transfer gradient:

- **Polytope Loss** (Exp1) should preferentially improve relational/logical probes (agreement, binding, argument structure), the tier that transfers cross-linguistically with a simple vocabulary bridge.
- **Sphere Loss** (Exp2) should preferentially improve perplexity and discourse-level coherence (next-token prediction in extended context, naturalness of generation), the tier that requires richer contextual representation and does not transfer via a vocabulary bridge.

This replaces an undirected 2×2 possibility matrix with a falsifiable directional prediction. If both loss functions affect the same probes equally, the VM4AI topology distinction does not map onto the BabyLM transfer gradient, and the topologies are not targeting distinct aspects of linguistic structure.

*Basis:* The BabyLM dict-axioms experiment (Wasserman, 2026, unpublished) shows that relational concepts transfer with rigid structure (vocabulary bridge) while discourse comprehension requires fluid contextual integration. VM4AI's Polytope (rigid) and Sphere (fluid) map onto these two tiers.

**H3 (French control):** French grammar accuracy under Sphere Loss will not exceed its exp8b baseline (87%).

**H4 (lambda scaling):** Higher lambda values will be required compared to Experiment 1. Constraining the full representation space is a stronger intervention than constraining attention distributions.

### 4.7 Success criteria

Same as Experiment 1.

### 4.8 Open design questions (to be resolved before registration)

1. Lambda range for Sphere Loss (Levin to specify)
2. Whether the norm constraint should apply to all layer outputs or specific layers
3. Whether NMA (Native Meaning Alignment) from VM4AI has a training-time analogue to incorporate

---

## 5. Experiment 3: Vector Invariance

### 5.1 Rationale

Experiments 1 and 2 ask whether we can synthesize the path to the destination. Experiment 3 asks whether the destination is the same across languages.

Levin (2025, VM4AI research notes) proposed that latent vectors for concepts may be model-invariant: a concept like "dog" should land at the same coordinate in latent space regardless of which model encodes it, because the vector represents a property of the concept rather than a property of the model. Levin's working hypothesis is that natural language input does not map 1:1 to latent vectors: there is a measurable distance between the surface form and the concept's "true" coordinate, and λ represents the pressure required to bridge that gap.

This hypothesis, if confirmed, would validate the foundational claim of the LLMs-as-telemetry framework: that what language models measure exists independently of any particular model, just as what a telescope measures exists independently of any particular telescope. It would also ground VM4AI's cognitive topologies in a universal substrate rather than model-specific artifacts.

The BabyLM dict-axioms result (Section 1.4) functions as a pilot study: a French model performing English entailment with only word translations is near-direct evidence that the latent geometry is shared. Experiment 3 tests this formally.

### 5.2 No new training required

This experiment uses existing exp8b checkpoints. All compute is analysis on already-trained models. The exp8b dataset provides the ideal test bed: 11 languages, identical architecture, identical hyperparameters, same seed, same training procedure, joint BPE tokenizer. All variables except language are held constant.

Estimated compute: <$10.

### 5.3 Concept set

A concept inventory of ~200 high-frequency concrete and abstract concepts that exist across all 11 languages, drawn from:

- Swadesh list (200 universal concepts)
- Concreteness norms (Brysbaert et al., 2014) for high-concreteness items
- Universal Dependencies treebank for cross-linguistic frequency validation

Exclusions: concepts with significant cross-linguistic lexicalization ambiguity, function words and grammatical particles, concepts with low frequency in any language's exp8b corpus.

The concept set will be partitioned into three tiers for H4 analysis:

| Tier | Examples | Predicted invariance |
|---|---|---|
| Relational/logical | negation, causation, entailment-adjacent concepts | Highest |
| Concrete/semantic | objects, actions, properties | Moderate |
| Discourse/pragmatic | emphasis, hedging, politeness markers | Lowest |

### 5.4 Alignment procedure

For each pair of languages (L1, L2):

1. Extract embeddings from the final hidden layer (post-LN, pre-head) of each language model at the final checkpoint (~380k steps)
2. For multi-token concepts, average the subword embeddings
3. Compute Procrustes alignment from L1 → L2 using a held-out anchor set of 50 concepts
4. Apply the learned rotation to the remaining ~150 test concepts
5. Measure cosine similarity between aligned L1 vectors and L2 vectors for test concepts
6. Compare against random baseline (shuffled concept assignments) and within-language baseline (cosine similarity of unrelated concept pairs)

### 5.5 Invariance score

For each language pair:

```
invariance(L1, L2) = mean_cosine_similarity(aligned_test_concepts) - random_baseline
```

Aggregate across all language pairs for a global invariance score, with 95% confidence intervals from bootstrap resampling of the concept set.

### 5.6 Hypotheses

**H0 (null):** After Procrustes alignment, latent vectors for matched concepts do not converge across languages. Mean cosine similarity between aligned concept embeddings remains <0.5 (consistent with random alignment after rotation).

**H1 (Levin's hypothesis):** After Procrustes alignment, latent vectors for matched concepts converge across languages. Mean cosine similarity exceeds 0.7 and significantly exceeds the within-language baseline computed from semantically unrelated word pairs.

**H2 (signal strength prediction):** Languages with higher WALS Agreement scores (predictors of training efficiency in exp8b) show tighter convergence to universal coordinates than languages with lower scores. Predicted: r > 0.5 correlation between morphological richness and convergence quality.

**H3 (synthetic language prediction):** Synthetic languages (synth_a through synth_d) fall along the WALS regression line, not as outliers. If vector invariance holds, even synthetic languages should converge on the same coordinates.

**H4 (concept-type stratification, from BabyLM transfer gradient):** Invariance scores will be non-uniform across concept types, stratified by the transfer gradient observed in the BabyLM dict-axioms experiment:

| Concept type | Predicted invariance | Basis |
|---|---|---|
| Relational/logical | Highest (>0.7) | Transferred cross-linguistically with only a vocabulary bridge |
| Concrete/semantic | Moderate (0.5–0.7) | Partial transfer in BabyLM; grounded in shared experience |
| Discourse/pragmatic | Lowest (<0.5) | No transfer in BabyLM; embedded in language-specific conventions |

This prediction refines H1 from a single mean threshold into a structured claim: the universal substrate is not uniform. Some concepts are more Platonic (language-independent) than others, and the hierarchy matches what the BabyLM transfer experiments independently reveal.

*Falsification:* If all concept types show equal invariance, the BabyLM transfer gradient does not reflect underlying geometric structure.

### 5.7 Pre-registered interpretation thresholds

| Global invariance score | Interpretation |
|---|---|
| <0.3 | H0 supported. Latent spaces are language-specific. Wittgensteinian reading stands. |
| 0.3–0.5 | Inconclusive. Some convergence but not strong. Partial invariance, possibly scale-dependent. |
| 0.5–0.7 | H1 partially supported. Significant convergence, but below the level Levin's hypothesis predicts. |
| >0.7 | H1 fully supported. Strong invariance. Platonic reading validated. |

---

## 6. Cross-Experiment Analysis Plan

The primary scientific contribution of this programme is not any individual experiment but the pattern across all three. The cross-experiment analysis tests the complete framework: universal destination (Exp3) + variable signal strength (exp8b baselines) + synthesizable path (Exp1, Exp2).

### 6.1 Topology-transfer mapping

The BabyLM transfer gradient predicts that VM4AI topologies map onto specific tiers of linguistic competence:

| VM4AI topology | Transfer tier | Exp1/Exp2 prediction | Exp3 prediction |
|---|---|---|---|
| Polytope (rigid/logic) | Relational/logical | Improves agreement, binding probes | Highest invariance for relational concepts |
| Sphere (fluid/creative) | Discourse/pragmatic | Improves perplexity, discourse coherence | Lowest invariance for discourse concepts |

This mapping is tested across all three experiments simultaneously. Confirmation requires consistent results across experiments; a single inconsistency falsifies the mapping.

### 6.2 Cross-experiment outcome matrix

| Exp1 result | Exp2 result | Exp3 result | Interpretation |
|---|---|---|---|
| No grammar improvement | No grammar improvement | No convergence | Strongest support for Language-Only Hypothesis: structure must be in the data, geometric manipulation cannot substitute, and latent spaces are language-specific |
| No grammar improvement | No grammar improvement | Strong convergence | Universal destination exists but geometric constraints cannot synthesize the path. Morphological structure is the only known route. |
| Grammar improvement (logical probes) | PPL improvement (discourse) | Strong convergence | Full confirmation of the framework: universal destination, distinct geometric paths to distinct tiers, synthesizable through topology-specific constraints |
| Grammar improvement (logical probes) | PPL improvement (discourse) | No convergence | Geometric constraints improve training dynamics but the resulting representations remain language-specific. Wittgensteinian acquisition with Wittgensteinian representation. |
| Both improve same probes | Both improve same probes | Any | VM4AI topology distinction does not map onto distinct linguistic dimensions. Topologies are redundant as training constraints. |

### 6.3 The Wittgenstein-Plato verdict

The synthesis proposed in Section 1.5 — Wittgenstein governs acquisition, Plato governs representation — requires:

1. **Exp8b baselines (already established):** Morphological structure determines acquisition efficiency. ✓
2. **Exp3 H1 supported:** Latent representations converge across languages (Platonic representation).
3. **Exp3 H4 supported:** Convergence is stratified by concept type, matching the BabyLM transfer gradient.
4. **Exp1/Exp2 H2 supported:** Different geometric topologies affect different tiers, matching the same gradient.

If all four hold, the synthesis is confirmed: different languages are different instruments measuring the same cognitive structure, with morphological richness determining instrument resolution, and VM4AI topologies providing a geometric vocabulary for describing both the instruments and the structure they measure.

If Exp3 fails (no convergence), the Platonic half collapses and we are left with a purely Wittgensteinian account: language constitutes cognition, and the measurement-instrument framing is metaphorical rather than literal.

Either outcome is scientifically valuable. The purpose of pre-registration is to ensure that the interpretation follows from the data, not the other way around.

---

## 7. Implications

### 7.1 For linguistics

If the telemetry framing holds — if LLM training dynamics reliably read out typological properties of language — then language models become a new class of fieldwork instrument. A linguist studying an under-documented language could train a small model on available text and read off morphological complexity, agreement richness, and structural density from the training curves, without explicit grammatical analysis. The exp8b correlations (r=-0.88 with WALS VerbSynth) suggest this is not speculative: the instrument already works, and the calibration curves already exist.

The Wittgenstein-Plato synthesis, if confirmed, adds a second capability: by testing cross-linguistic vector invariance, linguists could identify which semantic distinctions are universal (Platonic) and which are language-specific (Wittgensteinian), providing empirical traction on questions that have been debated philosophically for centuries.

### 7.2 For VM4AI

If Exp3 confirms vector invariance, VM4AI's cognitive topologies operate on a universal substrate rather than model-specific artifacts. The Topology Quantizer becomes a tool for navigating shared cognitive coordinates. If the topology-transfer mapping (Section 6.1) holds, it provides an empirical basis for why specific topologies produce specific cognitive effects: Polytope works for logic because logic is the tier of cognition with the strongest universal invariance; Sphere works for creativity because creative association operates in the tier most dependent on contextual richness.

### 7.3 For language model training

If Experiments 1 or 2 show that geometric constraints can replicate morphological advantage, this opens a practical path: instead of requiring morphologically rich training data, practitioners could apply topology-specific loss functions calibrated to the structural properties their task requires. The WALS-derived lambda parameterization provides a principled scaling: languages with low morphological complexity need higher geometric regularization.

### 7.4 For synthetic language design

The connection between geometric constraints (Levin) and morphological features (Wasserman) suggests that synthetic languages could be engineered to provide optimal training signal by design: languages whose morphological structure maps directly onto the geometric constraints that produce the best training dynamics. This connects to Wasserman's patent-pending work on morphological calibration and synthetic language training.

---

## 8. Collaboration and Intellectual Property

### 8.1 Contributions

**Edward Levin (VM4AI):**
- Polytope Loss and Sphere Loss concepts, derived from VM4AI cognitive topologies
- Topology Quantizer tool for lambda visualization and code generation
- BPE fertility parameterization and lambda sweet-spot analysis
- Vector invariance hypothesis (timestamped via VM4AI research notes)
- Core hypothesis: geometric training constraints can simulate morphological regularization

**Adam Wasserman:**
- Cross-linguistic baselines from 12-language controlled ablation (exp8b)
- WALS morphological parameterization (VerbSynth, Agreement, TAM, Fusion)
- BabyLM cross-linguistic transfer experiments (unpublished results motivating sharpened predictions)
- Experimental design, pre-registration framework, and analysis plan
- Training infrastructure and compute
- LLMs-as-telemetry theoretical framing

### 8.2 Intellectual property

- **VM4AI geometric engine and cognitive topologies:** Edward Levin & Karen Levin, CC-BY-NC-SA 4.0
- **Morphological calibration methods and per-language lambda tuning:** Subject to provisional patents held by Adam Wasserman
- **Joint experimental results:** Shared with attribution to both researchers

---

## 9. Timeline and Compute

| Phase | Experiments | Estimated compute | Duration |
|---|---|---|---|
| Pre-registration | File on OSF | — | Before first run |
| Exp1 training | 8 runs × 100k steps | ~2 days on 2× RTX 4090 | Week 1 |
| Exp2 training | 8 runs × 100k steps | ~2 days on 2× RTX 4090 | Week 2 |
| Exp3 analysis | Embedding extraction + alignment | <$10 (no training) | Week 1 (parallel) |
| Cross-experiment analysis | Sections 6.1–6.3 | CPU only | Week 3 |
| Write-up | — | — | Week 4 |

Experiment 3 can begin immediately (all checkpoints exist) and should be completed before Experiments 1 and 2 finish training, providing early evidence on the invariance hypothesis that contextualizes the training results.

---

## 10. References

Brysbaert, M., Warriner, A.B., & Kuperman, V. (2014). Concreteness ratings for 40 thousand generally known English word lemmas. *Behavior Research Methods*, 46, 904–911.

Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. *arXiv:2405.07987*.

Levin, E., & Levin, K. (2025). VM4AI: Virtual Machine for AI. CC-BY-NC-SA 4.0. https://vm4ai.com

Wasserman, A. (2026). Cross-linguistic training dynamics in small language models. Pre-registered on Open Science Framework. https://github.com/adamzwasserman/fractal-language

Wasserman, A. (2026). Born Speaking French: Cross-linguistic transfer in BabyLM. *Unpublished; submitted to BabyLM 2026.*

Wittgenstein, L. (1953). *Philosophical Investigations.* Blackwell.

---

*Pre-registration drafted: 2026-04-13*
*Adam Wasserman & Edward Levin (VM4AI)*
