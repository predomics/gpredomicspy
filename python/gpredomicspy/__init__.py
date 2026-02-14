"""
gpredomicspy - Python bindings for gpredomics
==============================================

Sparse, interpretable machine learning models using BTR (Binary/Ternary/Ratio)
languages. Built on a high-performance Rust engine with GPU acceleration.

Quick start::

    from gpredomicspy import Param, fit

    # Load parameters from YAML
    param = Param()
    param.load("param.yaml")
    param.set("max_epochs", 50)
    param.set("population_size", 5000)

    # Run the algorithm
    experiment = fit(param)

    # Inspect results
    best = experiment.best_population().best()
    print(f"Best model: AUC={best.get_metrics()['auc']:.4f}")
"""

from gpredomicspy._core import (
    Param,
    Individual,
    Population,
    Experiment,
    fit,
    filter_features,
    init_logger,
)

__version__ = "0.1.0"
__all__ = [
    "Param",
    "Individual",
    "Population",
    "Experiment",
    "fit",
    "filter_features",
    "init_logger",
]
