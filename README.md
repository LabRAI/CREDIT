# CREDIT

This repository provides the official implementation of the paper: _"CREDIT: Certified Ownership Verification of Deep Neural Networks Against Model Extraction Attacks"_

## Citation

If you find this work helpful, please consider citing our paper. We sincerely appreciate your support and interest in our work.

```bibtex
@misc{shen2026credit,
      title={CREDIT: Certified Ownership Verification of Deep Neural Networks Against Model Extraction Attacks}, 
      author={Bolin Shen and Zhan Cheng and Neil Zhenqiang Gong and Fan Yao and Yushun Dong},
      year={2026},
      eprint={2602.20419},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.20419}, 
}
```

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

**Output Example**:

```
[INFO] Running function: effectiveness()

{
    'baseline_name': 'CREDIT', 'baseline_model_name': 'resnet', 'tau': 100.0, 'auc': 1.0,
    'mi_sus_all': array([121.3, 118.7, 116.4,  84.2,  80.5,  79.1]),
    'sus_label': array([1, 1, 1, 0, 0, 0]),
    'pred_label': array([1, 1, 1, 0, 0, 0]), 'k': 5, 'n': 1000
}
```

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
