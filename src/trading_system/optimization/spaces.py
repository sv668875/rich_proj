"""Parameter search space utilities.

參數搜尋空間工具：描述不同優化方法可用的參數範圍與取值 | Define optimisation search spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    """Specification for a single optimisation parameter."""

    name: str
    param_type: str  # int | float | categorical
    grid: Optional[Sequence[float]] = None
    bounds: Optional[Tuple[float, float]] = None

    def ensure_valid(self) -> None:
        """Validate spec content."""

        if self.param_type not in {"int", "float", "categorical"}:
            raise ValueError(f"Unsupported parameter type: {self.param_type}")
        if self.param_type == "categorical" and not self.grid:
            raise ValueError("Categorical parameters must define grid values")
        if self.param_type in {"int", "float"} and self.bounds and self.bounds[0] >= self.bounds[1]:
            raise ValueError(f"Invalid bounds for {self.name}: {self.bounds}")


class ParameterSpace:
    """Container for strategy parameter search spaces."""

    def __init__(self, specs: Sequence[ParameterSpec]) -> None:
        self.specs = list(specs)
        for spec in self.specs:
            spec.ensure_valid()

    def grid_combinations(self) -> Iterator[Dict[str, float]]:
        """Yield cartesian combinations using provided grid values."""

        grids: List[Sequence[float]] = []
        names: List[str] = []

        for spec in self.specs:
            names.append(spec.name)
            if spec.grid:
                grids.append(spec.grid)
            elif spec.bounds:
                if spec.param_type == "int":
                    start, stop = int(spec.bounds[0]), int(spec.bounds[1])
                    grids.append(range(start, stop + 1))
                else:
                    grids.append(np.linspace(spec.bounds[0], spec.bounds[1], num=5))
            else:
                raise ValueError(f"Parameter {spec.name} requires either grid or bounds.")

        for combo in product(*grids):
            params = {}
            for name, value, spec in zip(names, combo, self.specs):
                params[name] = self._cast_value(value, spec)
            yield params

    def sample(self, n_samples: int, random_state: Optional[np.random.Generator] = None) -> List[Dict[str, float]]:
        """Randomly sample points within the defined space."""

        rng = random_state or np.random.default_rng()
        samples: List[Dict[str, float]] = []

        for _ in range(n_samples):
            params: Dict[str, float] = {}
            for spec in self.specs:
                params[spec.name] = self._sample_spec(spec, rng)
            samples.append(params)
        return samples

    def normalised_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Return bounds scaled to [0, 1] for numeric parameters."""

        bounds: Dict[str, Tuple[float, float]] = {}
        for spec in self.specs:
            if spec.param_type in {"int", "float"} and spec.bounds:
                bounds[spec.name] = spec.bounds
        return bounds

    @staticmethod
    def _cast_value(value: float, spec: ParameterSpec) -> float:
        if spec.param_type == "int":
            return int(round(float(value)))
        if spec.param_type == "categorical":
            return value
        return float(value)

    def _sample_spec(self, spec: ParameterSpec, rng: np.random.Generator) -> float:
        if spec.grid:
            choice = rng.choice(spec.grid)
            return self._cast_value(choice, spec)
        if spec.bounds:
            low, high = spec.bounds
            if spec.param_type == "int":
                return int(rng.integers(low, high + 1))
            return float(rng.uniform(low, high))
        raise ValueError(f"Parameter {spec.name} requires either grid or bounds.")


def get_default_space(strategy_name: str) -> ParameterSpace:
    """Return default parameter space for known strategies."""

    strategy_name = strategy_name.lower()
    if strategy_name == "adx_trend":
        specs = [
            ParameterSpec("ema_period", "int", grid=[150, 180, 200, 220, 250], bounds=(120, 260)),
            ParameterSpec("adx_threshold", "float", grid=[35.0, 45.0, 55.0, 65.0], bounds=(30.0, 70.0)),
            ParameterSpec("stoch_period", "int", grid=[10, 14, 18, 22], bounds=(8, 24)),
            ParameterSpec("risk_reward", "float", grid=[0.8, 1.0, 1.2, 1.5], bounds=(0.6, 1.8)),
        ]
        return ParameterSpace(specs)
    if strategy_name == "rsi_momentum":
        specs = [
            ParameterSpec("rsi_period", "int", grid=[4, 5, 6, 7, 8], bounds=(3, 12)),
            ParameterSpec("rsi_entry_threshold", "float", grid=[65.0, 70.0, 75.0, 80.0], bounds=(60.0, 85.0)),
            ParameterSpec("rsi_exit_threshold", "float", grid=[55.0, 60.0, 65.0, 70.0], bounds=(50.0, 75.0)),
        ]
        return ParameterSpace(specs)
    raise ValueError(f"尚未定義 {strategy_name} 的搜尋空間 | No search space for strategy {strategy_name}")
