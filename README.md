# Wasserman & Levin (2026)

## Can geometric manipulation of training dynamics replicate what natural language structure provides for free?

### The question

[Wasserman (2026)](https://github.com/adamzwasserman/fractal-language) showed that morphologically rich languages (French, Russian, Spanish) train dramatically more efficiently than morphologically poor languages (English, Chinese) on identical transformer architectures. French reaches grammatical competence at 197M tokens; English never reaches it through 4.3B tokens — a >20x efficiency gap driven entirely by linguistic structure.

This project tests whether **geometric constraints on training** — applied to the loss function, the attention mechanism, or the representation space — can close that gap artificially.

### The framework

Each experiment applies a different geometric lever and measures its impact against established linguistic baselines:

| Experiment | Geometric lever | What it constrains | Target axis |
|---|---|---|---|
| [Exp1: Polytope Loss](experiments/exp1_polytope/DESIGN.md) | Attention entropy minimization | Where the model looks | Behavioral |
| [Exp2: Sphere Loss](experiments/exp2_sphere/DESIGN.md) | Representation norm constraint | How the model organizes knowledge | Structural |

The baseline in every case is natural language morphology, measured via WALS features and cross-linguistic training results from [exp8b](https://github.com/adamzwasserman/fractal-language).

### The answer space

For each experiment, the outcome strengthens one of three interpretations:

- **No effect on grammar** — Strengthens the Language-Only Hypothesis: structure must be in the data, not the training dynamics. No amount of geometric manipulation can substitute for morphological signal.
- **Partial effect** — Geometric constraints approximate but don't fully replicate morphological advantage. The signal is partly in the data, partly in how the model processes it.
- **Full effect** — Morphological advantage can be synthesized through training dynamics. This connects directly to synthetic language design and morphological calibration methods.

### Pre-registrations

- Exp1 (Polytope): [TODO: file on OSF before first training run]
- Exp2 (Sphere): [TODO: file on OSF before first training run]

## Collaboration

- **Geometric loss functions (Polytope, Sphere), visualizer, and BPE fertility parameterization**: Edward Levin ([VM4AI](https://vm4ai.com))
- **WALS parameterization, experimental design, linguistic baselines, and training infrastructure**: Adam Wasserman
- **Morphological calibration methods**: Subject to provisional patents held by Adam Wasserman

## Structure

```
experiments/
  exp1_polytope/
    DESIGN.md              # Pre-registered design
    train_polytope.py      # Training script
  exp2_sphere/
    DESIGN.md              # Pre-registered design
tools/
  vm4ai/                   # Edward's VM4AI Topology Quantizer
results/                   # Training logs and analysis (generated)
```

## License

Code: MIT. Research outputs: CC-BY-4.0 with attribution to both authors.
