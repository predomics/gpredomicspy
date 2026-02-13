# gpredomicspy

Python bindings for the [gpredomics](https://github.com/predomics/gpredomics) Rust engine.

**gpredomics** discovers sparse, interpretable predictive models using BTR (Binary/Ternary/Ratio) languages. Designed for omics/metagenomics data but applicable to any binary classification task.

## Requirements

- Python >= 3.8
- Rust toolchain (for building from source)
- numpy, pandas

## Installation

```bash
# From source (requires Rust toolchain)
pip install .

# Or using maturin for development
pip install maturin
maturin develop
```

## Quick Start

```python
from gpredomicspy import Param, fit

# Load parameters
param = Param()
param.load("param.yaml")
param.set("max_epochs", 50)
param.set("population_size", 5000)

# Run
experiment = fit(param)

# Results
best = experiment.best_population().best()
metrics = best.get_metrics()
print(f"AUC: {metrics['auc']:.4f}")
print(f"Features: {best.get_features()}")
```

## Development Status

This package is in early development (Phase 1: Core Bindings).

## License

GPL-3.0 - see [gpredomics](https://github.com/predomics/gpredomics) for details.
