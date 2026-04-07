# Exp3: Vector Invariance

## Are Concept Coordinates Model-Independent?

## Pre-registration

**OSF registration**: [TODO: register before analysis begins]

## 1. Motivation

Wasserman (2026) argues that LLMs function as measurement instruments revealing cognitive structure already present in language, analogous to how telescopes reveal stars and microscopes reveal cells. The metaphor depends on a load-bearing assumption: that the thing being measured exists *independently* of any particular instrument.

Levin (2025, VM4AI research notes) independently proposed that latent vectors for concepts may be model-invariant: a concept like "dog" should land at the same coordinate in latent space regardless of which model encodes it, because the vector represents a property of the concept rather than a property of the model. Levin's working hypothesis is that natural language input does not map 1:1 to latent vectors. There is a measurable distance between the surface form and the concept's "true" coordinate, and the parameter λ represents the pressure required to bridge that gap.

This experiment tests both claims simultaneously:

1. **The invariance claim** (Levin): After alignment, do latent vectors for matched concepts converge across models trained on different languages?
2. **The instrument claim** (Wasserman): If invariance holds, the measurement-instrument metaphor has experimental grounding. If it fails, the metaphor is weaker than current framing suggests.

This is the missing experimental validation for the foundational thesis of Wasserman's research programme. It is also the first direct empirical test of Levin's vector invariance hypothesis.

## 2. Background

### The Platonic Representation Hypothesis

Huh et al. (2024) argue that as models scale, their internal representations converge toward a shared statistical model of reality. Different architectures and different training data produce increasingly similar latent geometries. This is consistent with Levin's hypothesis but currently demonstrated primarily at large scale (>1B parameters).

Exp3 tests whether the convergence holds at smaller scales (125M parameters) and across **language** rather than just architecture.

### Why this matters for Wasserman's programme

Wasserman's published cross-linguistic results are consistent with two competing interpretations:

| Interpretation | Claim | What it implies |
|---|---|---|
| Wittgensteinian | Language *constitutes* cognition. Different languages produce different cognitive structures. | The "measurement instrument" metaphor is poetic; what is happening is that the model is shaped by the language. |
| Platonic | Cognitive structure exists independently. Languages provide different signal strengths for revealing the same structure. | The "measurement instrument" metaphor is literal; different languages are different telescopes pointed at the same sky. |

Existing evidence from Wasserman (2026) does not discriminate between these. Exp3 is the discriminating experiment.

### Why this matters for Levin's framework

VM4AI's cognitive topologies operate on the assumption that there is a stable latent space to be navigated. If latent spaces are model-specific artifacts, then VM4AI's Polytope, Sphere, and other topologies are model-specific tools. If latent spaces are model-invariant, then VM4AI's topologies operate on a universal substrate, and the Topology Quantizer becomes a tool for navigating shared cognitive coordinates rather than per-model artifacts.

### Available data

The exp8b dataset (Wasserman, 2026) provides the ideal test bed:

- 11 languages (en, fr, es, fi, ru, vi, zh, synth_a, synth_b, synth_c, synth_d)
- Identical 125M GPT-2 architecture
- Identical hyperparameters, seed, training procedure
- Joint BPE tokenizer (50k vocab)
- Trained to ~380k steps (~780M tokens each)
- Checkpoints saved every 5k steps

This is the cleanest cross-linguistic dataset available for testing latent invariance. All variables except language are held constant.

## 3. Hypotheses

### H0 (null)
After Procrustes alignment, latent vectors for matched concepts do NOT converge across languages. Mean cosine similarity between aligned concept embeddings remains <0.5 (consistent with random alignment after rotation).

### H1 (Levin's hypothesis)
After Procrustes alignment, latent vectors for matched concepts converge across languages. Mean cosine similarity between aligned concept embeddings exceeds 0.7 (significantly above random) and significantly above the within-language baseline computed from semantically unrelated word pairs.

### H2 (signal strength prediction)
Languages with higher WALS Agreement scores (predictors of training efficiency in exp8b) will show *tighter* convergence to the universal coordinates than languages with lower WALS Agreement. This connects vector invariance to morphological signal strength: better instruments produce sharper measurements.

### H3 (synthetic language prediction)
Synthetic languages (synth_a-synth_d) will fall *along the WALS axis*, not as outliers. If vector invariance holds, even synthetic languages should converge on the same coordinates, supporting the claim that the universal substrate is independent of any particular linguistic surface form.

## 4. Experimental Design

### 4.1 No new training required

This experiment uses existing exp8b checkpoints. All compute is analysis on already-trained models. Estimated cost: <$10 for cross-checkpoint analysis.

### 4.2 Concept set

Construct a concept inventory of ~200 high-frequency concrete and abstract concepts that exist across all 11 languages. Sources:

- Swadesh list (200 universal concepts)
- Concreteness norms (Brysbaert et al. 2014) for high-concreteness items
- Universal Dependencies treebank for cross-linguistic frequency validation

For each concept, identify the canonical translation in each language. Excludes:
- Concepts with significant cross-linguistic ambiguity (e.g., emotion words known to lexicalize differently)
- Function words and grammatical particles
- Concepts with low frequency in any language's exp8b corpus

### 4.3 Embedding extraction

For each language model (final checkpoint, ~380k steps):

1. Tokenize each canonical translation
2. Extract the embedding from the final hidden layer (post-LN, pre-head)
3. For multi-token concepts, average the subword embeddings
4. Build a (n_concepts × d_model) matrix per language

### 4.4 Alignment procedure

For each pair of languages (L1, L2):

1. Compute Procrustes alignment from L1 → L2 using a held-out **anchor set** of 50 concepts
2. Apply the learned rotation to the remaining 150 concepts
3. Measure cosine similarity between aligned L1 vectors and L2 vectors for the test concepts
4. Compare against:
   - Random baseline (shuffled concept assignments)
   - Within-language baseline (cosine similarity of unrelated concept pairs in L2)

### 4.5 Cross-linguistic invariance score

For each language pair (L1, L2):

```
invariance(L1, L2) = mean_cosine_similarity(aligned_test_concepts) - random_baseline
```

Aggregate across all language pairs to compute a global invariance score.

### 4.6 WALS correlation analysis

Test H2: Does WALS Agreement score predict tighter convergence?

For each language, compute its mean invariance score across all pairings. Correlate with WALS composite (22A + 29A + 21B + 20A) from exp8b. Predicted: r > 0.5 (positive correlation between morphological richness and convergence quality).

### 4.7 Synthetic language analysis

Test H3: Do synth_a-synth_d behave as predicted by their WALS scores?

Plot all 11 languages on (WALS composite, invariance score) axes. If H3 is correct, synthetic languages should fall along the regression line, not as outliers.

## 5. Analysis Plan

### Primary analysis
Mean cross-linguistic invariance score after Procrustes alignment, with 95% confidence intervals from bootstrap resampling of the concept set.

### Pre-registered thresholds

| Score range | Interpretation |
|---|---|
| <0.3 | H0 supported. Latent spaces are language-specific. Wittgensteinian reading of Wasserman's results stands. |
| 0.3 - 0.5 | Inconclusive. Some convergence but not strong. Suggests partial invariance, possibly scale-dependent. |
| 0.5 - 0.7 | H1 partially supported. Significant convergence above random, but not at the level Levin's hypothesis predicts. |
| >0.7 | H1 fully supported. Strong invariance across languages. Platonic reading of Wasserman's results validated. |

### Secondary analyses
1. WALS correlation (H2)
2. Synthetic language behavior (H3)
3. Concept-level analysis: which concepts show strongest invariance? Which are most language-specific?
4. Layer-wise analysis: does invariance increase or decrease across model layers?

### Reporting commitment
All results reported regardless of outcome. Null results are scientifically valuable: they constrain both Levin's vector invariance hypothesis and Wasserman's measurement-instrument framing. This experiment is structured to be informative under any outcome.

## 6. Implications

### If H1 is supported

This is the experimental validation Wasserman's programme has been missing. The measurement-instrument metaphor is no longer poetic; it has empirical grounding. Different languages are different telescopes pointed at the same cognitive sky, and Exp3 demonstrates that the sky exists independently of the telescope.

For Levin's framework, vector invariance means VM4AI's cognitive topologies operate on a universal substrate. The Topology Quantizer becomes a tool for navigating shared coordinates rather than model-specific artifacts.

The combined Wasserman-Levin framework becomes a complete two-layer theory:

| Layer | Claim |
|---|---|
| Universal substrate | Concepts and grammatical abstractions exist as model-invariant coordinates |
| Linguistic surface | Vocabulary and morphology determine signal strength for reaching the substrate |

This framework has direct implications for synthetic language design (Wasserman's patent work): a language can be engineered to map optimally onto the universal coordinates, minimizing the distance between surface form and conceptual destination.

### If H1 is rejected

Equally informative. We learn that latent invariance does not hold at 125M scale, which constrains both the Platonic Representation Hypothesis and Levin's working hypothesis to larger model scales. Wasserman's existing programme is not weakened. It just retains the current framing of language structure as the locus of measurement.

The scientific value is in either direction.

### For the BabyLM submission (Wasserman 2026)

If H1 is supported, the BabyLM submission gains a striking secondary finding: the French-trained model converges on the same conceptual coordinates as English-trained models, even though it never saw English. This transforms the BabyLM argument from "grammar transfers cross-linguistically" to "the French-trained model arrives at the same conceptual destination as English-trained models, validating that LLM training reveals universal cognitive structure rather than language-specific artifacts."

## 7. Collaboration

### Edward Levin ([VM4AI](https://vm4ai.com))
- Vector invariance hypothesis (the core empirical claim being tested)
- Theoretical framework for the lambda parameter as "pressure required to bridge NL word and its latent vector"
- Independent line of research on ZHK (zero human knowledge) latent analysis

### Adam Wasserman ([fractal-language](https://github.com/adamzwasserman/fractal-language))
- Cross-linguistic checkpoint dataset (exp8b, 11 languages)
- WALS feature data and morphological analysis
- Connection to the measurement-instrument framing of the Language-Only Hypothesis
- Experimental design and pre-registration

### Joint
- Analysis methodology
- Interpretation of results
- Potential joint publication

### Intellectual property
- Vector invariance hypothesis: Edward Levin (timestamped via VM4AI research notes)
- Exp8b cross-linguistic dataset and WALS framework: Adam Wasserman
- Joint experimental results: Shared with attribution to both researchers

## 8. Relation to Other Experiments

| Experiment | What it tests | How Exp3 relates |
|---|---|---|
| Exp1 (Polytope Loss) | Can attention entropy minimization simulate morphological signal? | Tests whether geometric constraints can substitute for linguistic structure |
| Exp2 (Sphere Loss) | Can representation norm constraints simulate morphological signal? | Tests whether representation geometry can substitute for linguistic structure |
| **Exp3 (Vector Invariance)** | **Do all languages converge on the same latent coordinates?** | **Tests whether the substrate being measured is universal** |

Exp1 and Exp2 ask whether we can synthesize the *path* to the destination. Exp3 asks whether the *destination* is the same across languages. Together, they test the complete framework: universal destination + variable signal strength + synthesizable path.

## 9. Compute estimate

**Total: <$10**

- No model training required
- Embedding extraction: <1 hour on RTX 4090 across all 11 models × all checkpoints
- Procrustes alignment + analysis: minutes on CPU
- Bootstrap resampling: <1 hour on CPU

## 10. Timeline

This experiment can be completed in days, not weeks. The bottleneck is concept set curation (Section 4.2), not compute. Pre-registration should happen within 1 week of design finalization.

---

*Design drafted: 2026-04-07*
*Adam Wasserman + Edward Levin (VM4AI)*
