# CREDIT

This repository provides the official implementation of the paper: _"CREDIT: Certified Ownership Verification of Deep Neural Networks Against Model Extraction Attacks"_

## Quick Start

We use `uv` to manage Python environments and dependencies, which offers the most convenient and reproducible way to set
up the project.

```
# Install dependencies
uv sync
```

## Experiments

Run experiments by calling functions defined in main.py.

**Effectiveness**: Evaluates CREDIT against baselines under model extraction scenarios.

```shell
# Run effectiveness evaluation
python main.py effectiveness
```

**Efficiency**: Measures runtime overhead of defense and verification.

```shell
# Run efficiency evaluation
python main.py efficiency
```

**Parameters**: Studies the impact of parameters (e.g., β, τ, σ) on verification robustness.

```shell
# Run parameter sensitivity experiments
python main.py parameters

```

**Plots**: Visualization of experimental results (e.g., sigma–gamma curves, optimal ranges, parameter bands).

```shell
# Generate plots
python main.py plots
```

Before running experiments, make sure you have already trained:

- a target model (the protected model)
- an independent model (a reference model trained independently)

With these prepared, CREDIT can be deployed seamlessly for verification and defense evaluation.

## Project Structure

```md
.
├── main.py # Entry script for running experiments
├── pyproject.toml # Project dependencies managed by uv
├── models/ # CREDIT method and baseline implementations
├── utils/ # Utility functions
├── experiments/ # Experiment pipelines (effectiveness, efficiency, parameters)
└── README.md # Project documentation
```

## License

MIT License
