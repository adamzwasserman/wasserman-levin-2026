# Wasserman & Levin (2026): Polytope Loss Experiments

Can attention regularization synthesize morphological advantage in LLM training?

## Background

[Wasserman (2026)](https://github.com/adamzwasserman/fractal-language) showed that morphologically rich languages (French, Russian, Spanish) train dramatically more efficiently than morphologically poor languages (English, Chinese) on identical transformer architectures. English grammar accuracy remains at chance level (40%) through 4.3B tokens while French reaches 100% at 197M tokens.

This repository tests whether the **Polytope Loss** — an attention entropy penalty — can break the English grammar ceiling by simulating the implicit regularization that morphological structure provides naturally.

## Collaboration

- **Polytope Loss concept, visualizer, and BPE fertility parameterization**: Edward Levin ([VM4AI](https://vm4ai.com))
- **WALS parameterization, experimental design, baselines, and training infrastructure**: Adam Wasserman
- **Morphological calibration methods**: Subject to provisional patents held by Adam Wasserman

## Experiment

See [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) for the full pre-registered design.

**Pre-registration**: [TODO: file on OSF before first training run]

### Summary

8 training runs testing two parameterizations of lambda (BPE fertility vs WALS morphological features) on English and French 125M transformers, benchmarked against established baselines from the cross-linguistic scaling experiment.

**Primary prediction (H0)**: The attention entropy penalty will NOT break the English grammar ceiling, because the deficit is structural — it exists in the data, not the training dynamics.

## Structure

```
EXPERIMENT_DESIGN.md   # Pre-registered experimental design
train_polytope.py      # Training script with Polytope Loss
tools/                 # Edward's VM4AI Topology Quantizer
results/               # Training logs and analysis (generated)
```

## License

Code: MIT. Research outputs: CC-BY-4.0 with attribution to both authors.
