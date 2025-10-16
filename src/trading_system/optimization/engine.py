"""Strategy optimisation engine.

策略最佳化引擎：提供網格搜尋與簡易貝式最佳化，並在回測執行後回傳績效紀錄 | Optimise strategies
using grid search and a lightweight Bayesian optimiser backed by Gaussian Processes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..backtest import run_backtest
from ..config_loader import StrategyConfig
from .spaces import ParameterSpace, get_default_space

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency
    optuna = None


EvaluationCallback = Callable[[int, Dict[str, float], Dict[str, float]], None]


@dataclass
class OptimisationMetadata:
    """Metadata associated with an optimisation run."""

    strategy_name: str
    method: str
    started_at: datetime
    iterations: int
    options: Dict[str, Any]
    random_seed: Optional[int]


class StrategyOptimizer:
    """Coordinate optimisation routines for strategies."""

    def __init__(
        self,
        base_config: StrategyConfig,
        strategy_name: str,
        lookback_days: int,
        initial_equity: float,
        parameter_space: Optional[ParameterSpace] = None,
        on_evaluation: Optional[EvaluationCallback] = None,
        method_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.base_config = base_config
        self.strategy_name = strategy_name
        self.lookback_days = lookback_days
        self.initial_equity = initial_equity
        self.space = parameter_space or get_default_space(strategy_name)
        self.on_evaluation = on_evaluation
        self.method_kwargs = dict(method_kwargs or {})

    def optimise(
        self,
        method: str,
        iterations: int,
        random_seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, OptimisationMetadata]:
        started_at = datetime.utcnow()
        method = method.lower()
        if method == "grid":
            results = self._grid_search(max_iterations=iterations)
        elif method == "bayesian":
            results = self._bayesian_search(iterations=iterations, random_seed=random_seed)
        elif method == "optuna":
            results = self._optuna_search(iterations=iterations, random_seed=random_seed)
        elif method in {"rl", "reinforcement"}:
            results = self._reinforcement_search(iterations=iterations, random_seed=random_seed)
        else:
            raise ValueError(f"不支援的最佳化方法 {method} | Unsupported optimisation method {method}")

        metadata = OptimisationMetadata(
            strategy_name=self.strategy_name,
            method=method,
            started_at=started_at,
            iterations=len(results),
            options=dict(self.method_kwargs),
            random_seed=random_seed,
        )
        df = pd.DataFrame(results)
        df["method"] = method
        df["strategy_name"] = self.strategy_name
        df["started_at"] = metadata.started_at.isoformat()
        if not df.empty:
            df["method_options"] = [dict(self.method_kwargs) for _ in range(len(df))]
        df.sort_values(["total_return_pct", "win_rate"], ascending=[False, False], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df, metadata

    def _grid_search(self, max_iterations: int) -> List[Dict[str, float]]:
        results: List[Dict[str, float]] = []
        for iteration, params in enumerate(self.space.grid_combinations(), start=1):
            if iteration > max_iterations:
                break
            metrics = self._evaluate_parameters(params)
            record = self._format_record(iteration, params, metrics)
            results.append(record)
            if self.on_evaluation:
                self.on_evaluation(iteration, params, metrics)
        return results

    def _bayesian_search(self, iterations: int, random_seed: Optional[int]) -> List[Dict[str, float]]:
        rng = np.random.default_rng(random_seed)
        bounds = self.space.normalised_bounds()
        if not bounds:
            raise ValueError("貝式最佳化需要連續型參數範圍 | Bayesian optimisation requires numeric bounds.")

        evaluated_params: List[Dict[str, float]] = []
        evaluated_scores: List[float] = []
        evaluated_metrics: List[Dict[str, float]] = []
        initial_samples = min(5, iterations)
        samples = self.space.sample(initial_samples, random_state=rng)

        results: List[Dict[str, float]] = []

        for iteration in range(1, iterations + 1):
            if iteration <= len(samples):
                params = samples[iteration - 1]
            else:
                params = self._suggest_next(bounds, evaluated_params, evaluated_scores, rng)

            metrics = self._evaluate_parameters(params)
            target = float(metrics.get("total_return_pct", 0.0))
            evaluated_params.append(params)
            evaluated_scores.append(target)
            evaluated_metrics.append(metrics)

            record = self._format_record(iteration, params, metrics)
            results.append(record)
            if self.on_evaluation:
                self.on_evaluation(iteration, params, metrics)
        return results

    def _optuna_search(self, iterations: int, random_seed: Optional[int]) -> List[Dict[str, float]]:
        if optuna is None:
            raise RuntimeError(
                "Optuna 尚未安裝，請執行 `pip install optuna` 後再試。 | "
                "Optuna is not installed. Run `pip install optuna` to enable this optimiser."
            )

        sampler_name = str(self.method_kwargs.get("optuna_sampler", "tpe")).lower()
        if sampler_name == "cmaes":
            sigma0 = self.method_kwargs.get("optuna_sigma0")
            if sigma0 is None:
                sampler = optuna.samplers.CmaEsSampler(seed=random_seed)
            else:
                sampler = optuna.samplers.CmaEsSampler(seed=random_seed, sigma0=float(sigma0))
        elif sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed=random_seed)
        else:
            sampler = optuna.samplers.TPESampler(
                seed=random_seed,
                n_startup_trials=int(self.method_kwargs.get("optuna_startup_trials", 10)),
            )
        pruner_name = str(self.method_kwargs.get("optuna_pruner", "none")).lower()
        if pruner_name == "median":
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=int(self.method_kwargs.get("optuna_pruner_startup", 5)),
                n_warmup_steps=int(self.method_kwargs.get("optuna_pruner_warmup", 0)),
            )
        else:
            pruner = None

        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        interim_results: List[Dict[str, float]] = []

        def objective(trial: "optuna.Trial") -> float:
            params = {}
            for spec in self.space.specs:
                if spec.param_type == "int":
                    if spec.grid:
                        params[spec.name] = trial.suggest_categorical(spec.name, list(spec.grid))
                    elif spec.bounds:
                        low, high = int(spec.bounds[0]), int(spec.bounds[1])
                        params[spec.name] = trial.suggest_int(spec.name, low, high)
                    else:
                        raise ValueError(f"{spec.name} 需定義 grid 或 bounds | Missing grid/bounds for {spec.name}")
                elif spec.param_type == "float":
                    if spec.grid:
                        params[spec.name] = trial.suggest_categorical(spec.name, list(spec.grid))
                    elif spec.bounds:
                        low, high = float(spec.bounds[0]), float(spec.bounds[1])
                        params[spec.name] = trial.suggest_float(spec.name, low, high)
                    else:
                        raise ValueError(f"{spec.name} 需定義 grid 或 bounds | Missing grid/bounds for {spec.name}")
                else:
                    if not spec.grid:
                        raise ValueError(f"{spec.name} 的類別型參數需提供 grid | Categorical {spec.name} needs grid.")
                    params[spec.name] = trial.suggest_categorical(spec.name, list(spec.grid))

            metrics = self._evaluate_parameters(params)
            record = self._format_record(len(interim_results) + 1, params, metrics)
            interim_results.append(record)
            if self.on_evaluation:
                self.on_evaluation(len(interim_results), params, metrics)
            return float(metrics.get("total_return_pct", 0.0))

        study.optimize(
            objective,
            n_trials=iterations,
            show_progress_bar=False,
            n_jobs=int(self.method_kwargs.get("optuna_jobs", 1)),
        )
        return interim_results

    def _reinforcement_search(self, iterations: int, random_seed: Optional[int]) -> List[Dict[str, float]]:
        rng = np.random.default_rng(random_seed)
        pool_multiplier = max(int(self.method_kwargs.get("rl_pool_multiplier", 4)), 1)
        candidate_pool = self.space.sample(max(iterations * pool_multiplier, 20), random_state=rng)
        q_values: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        key_to_params: Dict[str, Dict[str, float]] = {}
        epsilon = float(self.method_kwargs.get("rl_epsilon_start", 0.3))
        epsilon_min = float(self.method_kwargs.get("rl_epsilon_min", 0.05))
        decay = float(self.method_kwargs.get("rl_epsilon_decay", 0.95))
        reward_mix = float(self.method_kwargs.get("rl_reward_mix", 0.6))
        results: List[Dict[str, float]] = []

        def _param_key(param_dict: Dict[str, float]) -> str:
            return repr(sorted(param_dict.items()))

        for iteration in range(1, iterations + 1):
            explore = rng.random() < epsilon or not q_values
            if explore:
                params = dict(candidate_pool[rng.integers(0, len(candidate_pool))])
            else:
                best_key = max(q_values.items(), key=lambda item: item[1])[0]
                params = dict(key_to_params[best_key])
            metrics = self._evaluate_parameters(params)
            key = _param_key(params)
            key_to_params[key] = params
            record = self._format_record(iteration, params, metrics)
            total_return_pct = float(record.get("total_return_pct", 0.0))
            drawdown_pct = float(record.get("max_drawdown_pct", 0.0))
            win_rate = float(record.get("win_rate", 0.0))
            reward = (reward_mix * total_return_pct) + ((1 - reward_mix) * win_rate) - drawdown_pct
            counts[key] = counts.get(key, 0) + 1
            q_prev = q_values.get(key, 0.0)
            alpha = 1.0 / counts[key]
            q_values[key] = q_prev + alpha * (reward - q_prev)

            results.append(record)
            if self.on_evaluation:
                self.on_evaluation(iteration, params, metrics)
            epsilon = max(epsilon * decay, epsilon_min)

        results.sort(key=lambda r: (r.get("total_return_pct", 0.0), r.get("win_rate", 0.0)), reverse=True)
        for idx, record in enumerate(results, start=1):
            record["iteration"] = idx
        return results

    def _suggest_next(
        self,
        bounds: Dict[str, Tuple[float, float]],
        evaluated_params: List[Dict[str, float]],
        evaluated_scores: List[float],
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        X = np.array([self._to_vector(p, bounds) for p in evaluated_params])
        y = np.array(evaluated_scores, dtype=float)
        gp = _GaussianProcessRegressor(length_scale=1.6, noise=1e-6)
        gp.fit(X, y)

        candidate_pool = self.space.sample(128, random_state=rng)
        best_candidate = candidate_pool[0]
        best_score = -float("inf")
        best_y = float(np.max(y)) if y.size else 0.0

        for candidate in candidate_pool:
            vector = self._to_vector(candidate, bounds)
            mu, sigma = gp.predict(vector)
            ei = _expected_improvement(mu, sigma, best_y)
            if ei > best_score:
                best_score = ei
                best_candidate = candidate

        return best_candidate

    def _evaluate_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        runtime_config = StrategyConfig(
            metadata=self.base_config.metadata,
            market=self.base_config.market,
            risk=self.base_config.risk,
            data=self.base_config.data,
            parameters=dict(params),
        )
        try:
            results = run_backtest(
                strategy_name=self.strategy_name,
                config=runtime_config,
                lookback_days=self.lookback_days,
                persist=False,
                initial_equity=self.initial_equity,
            )
            metrics = results.get("metrics", {}) or {}
        except Exception as exc:  # pylint: disable=broad-except
            metrics = {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 1.0,
                "total_return": 0.0,
                "num_trades": 0,
                "total_return_pct": 0.0,
                "initial_equity": self.initial_equity,
                "ending_equity": self.initial_equity,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "error": str(exc),
            }
        return metrics

    def _format_record(
        self,
        iteration: int,
        params: Dict[str, float],
        metrics: Dict[str, float],
    ) -> Dict[str, float]:
        record = dict(metrics)
        record["iteration"] = iteration
        record["params"] = params
        max_drawdown = float(metrics.get("max_drawdown", 0.0))
        initial_equity = float(metrics.get("initial_equity", self.initial_equity or 1.0))
        record["max_drawdown_pct"] = max_drawdown / initial_equity if initial_equity else 0.0
        return record

    def _to_vector(self, params: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> np.ndarray:
        vector: List[float] = []
        for spec in self.space.specs:
            value = float(params[spec.name])
            if spec.name in bounds:
                low, high = bounds[spec.name]
                if math.isclose(high, low):
                    vector.append(0.5)
                else:
                    vector.append((value - low) / (high - low))
            else:
                vector.append(value)
        return np.array(vector, dtype=float)


class _GaussianProcessRegressor:
    """Minimal Gaussian Process regressor with RBF kernel."""

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-6) -> None:
        self.length_scale = length_scale
        self.noise = noise
        self.X_train: Optional[np.ndarray] = None
        self.alpha: Optional[np.ndarray] = None
        self.K_inv: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.size == 0:
            self.X_train = np.empty((0, 0))
            self.alpha = None
            self.K_inv = None
            return

        self.X_train = X
        self.y_train = y
        K = self._kernel(X, X)
        K += self.noise * np.eye(K.shape[0])
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            jitter = 1e-8
            self.K_inv = np.linalg.inv(K + jitter * np.eye(K.shape[0]))
        self.alpha = self.K_inv @ y

    def predict(self, x: np.ndarray) -> Tuple[float, float]:
        if self.X_train is None or self.X_train.size == 0 or self.alpha is None or self.K_inv is None:
            return 0.0, 1.0
        k_star = self._kernel(self.X_train, x[None, :]).reshape(-1, 1)
        mu = float(k_star.T @ self.alpha)
        k_ss = float(self._kernel(x[None, :], x[None, :]))
        sigma = k_ss - float(k_star.T @ self.K_inv @ k_star)
        sigma = max(sigma, 1e-6)
        return mu, math.sqrt(sigma)

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if X1.size == 0 or X2.size == 0:
            return np.zeros((X1.shape[0], X2.shape[0]))
        sqdist = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        factor = -0.5 / (self.length_scale**2)
        return np.exp(factor * sqdist)


def _expected_improvement(mean: float, sigma: float, best_y: float, xi: float = 0.01) -> float:
    """Compute expected improvement acquisition value."""

    if sigma <= 0:
        return 0.0
    z = (mean - best_y - xi) / sigma
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
    pdf = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z**2)
    improvement = (mean - best_y - xi) * cdf + sigma * pdf
    return max(improvement, 0.0)
