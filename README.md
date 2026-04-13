# Wasserman & Levin (2026)

## Can geometric manipulation of training dynamics replicate what natural language structure provides for free?

### The question

Two independent lines of research converge here:

**Wasserman (2026)** showed that morphologically rich languages (French, Russian, Spanish) train dramatically more efficiently than morphologically poor languages (English, Chinese) on identical transformer architectures. French reaches grammatical competence at 197M tokens; English never reaches it through 4.3B tokens, a >20x efficiency gap driven entirely by linguistic structure. The question is whether this gap is inherent to the data or can be closed.

**Levin & Levin (2025)** developed [VM4AI](https://vm4ai.com) (Virtual Machine for AI), a geometry-aware cognitive controller that enforces specific topological constraints — Polytope, Sphere, Torus, Mobius, and others — on AI reasoning at inference time. VM4AI demonstrates that geometric structure shapes cognitive output: a Polytope topology produces rigid logical reasoning, while a Sphere topology enables fluid creative association. The framework operates as a client-side overlay across multiple AI platforms (OpenAI, Anthropic, xAI, Google, Microsoft).

This project asks: **if geometric constraints shape AI cognition at inference time, can they also shape learning at training time?** Specifically, can the Polytope and Sphere geometries from VM4AI, applied as loss function constraints during pretraining, replicate the structural advantages that morphologically rich languages provide naturally?

### The framework

Each experiment takes a geometric topology from VM4AI and translates it into a training-time loss constraint, measuring its impact against established cross-linguistic baselines:

| Experiment | VM4AI Topology | Training-time translation | What it constrains |
|---|---|---|---|
| [Exp1: Polytope Loss](experiments/exp1_polytope/DESIGN.md) | Polytope (Rigid/Logic) | Attention entropy minimization | Where the model looks |
| [Exp2: Sphere Loss](experiments/exp2_sphere/DESIGN.md) | Sphere (Fluid/Creative) | Representation norm constraint | How the model organizes knowledge |
| [Exp3: Vector Invariance](experiments/exp3_invariance/DESIGN.md) | (Cross-cutting) | Cross-linguistic latent alignment analysis | Whether the substrate being measured is universal |

The baseline in every case is natural language morphology, measured via WALS features and cross-linguistic training results from [Wasserman's 12-language experiment](https://github.com/adamzwasserman/fractal-language).

### The answer space

For each experiment, the outcome strengthens one of three interpretations:

- **No effect on grammar**: Strengthens the Language-Only Hypothesis: structure must be in the data, not the training dynamics. No amount of geometric manipulation can substitute for morphological signal.
- **Partial effect**: Geometric constraints approximate but don't fully replicate morphological advantage. The signal is partly in the data, partly in how the model processes it.
- **Full effect**: Morphological advantage can be synthesized through training dynamics, connecting VM4AI's inference-time geometry to training-time optimization and opening the door to engineered linguistic structure for accelerated model training.

### Pre-registrations

- Exp1 (Polytope): [TODO: file on OSF before first training run]
- Exp2 (Sphere): [TODO: file on OSF before first training run]
- Exp3 (Vector Invariance): [TODO: file on OSF before analysis begins]

## Researchers

### Edward Levin (VM4AI)

Edward Levin is the architect of [VM4AI](https://vm4ai.com) (Virtual Machine for AI), a geometry-aware cognitive controller that applies topological constraints to structure AI reasoning. VM4AI defines eight cognitive topologies — Polytope, Sphere, Torus, Hive, Grid, Fork, Flux, and Mobius — each imposing a distinct geometric "shape of logic" on AI output. The framework includes Native Meaning Alignment (NMA) for semantic precision, a Hive architecture for simulated multi-agent consensus, and a Continuity Protocol for cross-platform state portability via MCP.

VM4AI was developed by Edward Levin and Karen Levin and is published under CC-BY-NC-SA 4.0. It is currently in Public Pilot (v9.9.28).

**Contributions to this project**: Geometric loss function design (Polytope, Sphere), the Topology Quantizer tool, BPE fertility parameterization, and the lambda sweet-spot analysis. Edward proposed the core hypothesis that geometric constraints applied during training could simulate the implicit regularization observed in morphologically rich languages.

Contact: contact@vm4ai.com

### Adam Wasserman

Adam Wasserman is an independent researcher studying cross-linguistic training dynamics in language models. His controlled ablation experiments — training identical transformers on matched corpora across 12 languages while holding every variable constant except the training language — established that morphological structure is the primary driver of training efficiency, not scale. These results, pre-registered on the Open Science Framework, falsify the universality assumption of the scaling hypothesis.

**Contributions to this project**: Cross-linguistic baselines (12 languages, WALS morphological features), experimental design and pre-registration, WALS-derived lambda parameterization, and training infrastructure. Adam's data provides the ground truth that geometric interventions are measured against.

Contact: wassermana@gmail.com | [fractal-language](https://github.com/adamzwasserman/fractal-language)

### Intellectual property

- **VM4AI cognitive topologies and geometric engine**: Edward Levin & Karen Levin, CC-BY-NC-SA 4.0
- **Joint experimental results**: Shared, with attribution to both researchers

## Structure

```
experiments/
  exp1_polytope/
    DESIGN.md              # Pre-registered design
    train_polytope.py      # Training script
  exp2_sphere/
    DESIGN.md              # Pre-registered design
tools/
  vm4ai/                   # VM4AI Topology Quantizer
results/                   # Training logs and analysis (generated)
```

## License

Code: MIT. Research outputs: CC-BY-4.0 with attribution to both researchers. VM4AI tools included under their original CC-BY-NC-SA 4.0 license.
