"""Optimisation package exports."""

from .engine import StrategyOptimizer, OptimisationMetadata
from .spaces import ParameterSpace, ParameterSpec, get_default_space

__all__ = [
    "StrategyOptimizer",
    "OptimisationMetadata",
    "ParameterSpace",
    "ParameterSpec",
    "get_default_space",
]
