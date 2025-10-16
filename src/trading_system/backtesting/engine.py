"""Backtesting engine skeleton.

回測引擎骨架：管理策略執行與績效計算 | Skeleton engine to run strategies and compute metrics.
"""

from dataclasses import dataclass
from typing import Dict, Type

import pandas as pd

from ..config_loader import StrategyConfig
from ..strategies.base import Strategy


@dataclass
class BacktestResult:
    """統一儲存回測結果 | Container for backtest results."""

    equity: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    total_return: float
    num_trades: int

    def as_dict(self) -> Dict[str, float]:
        """Convert to serializable dictionary."""

        return {
            "equity": self.equity,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "total_return": self.total_return,
            "num_trades": self.num_trades,
        }


class BacktestEngine:
    """Orchestrate backtest loop.

    管理策略回測流程：資料迭代、交易信號與績效統計 | Handle data iteration and metric aggregation.
    """

    def __init__(self, strategy_cls: Type[Strategy], config: StrategyConfig) -> None:
        self.strategy_cls = strategy_cls
        self.config = config

    def run(self, candles: pd.DataFrame, initial_equity: float = 1.0) -> Dict[str, float]:
        """Execute strategy over OHLCV data.

        在OHLCV資料上運行策略並產出績效指標 | Iterate over candles to produce performance stats.
        """

        strategy = self.strategy_cls(config=self.config)
        trade_log, equity_curve = strategy.simulate(candles, initial_equity=initial_equity)
        metrics = self._compute_metrics(
            trade_log=trade_log,
            equity_curve=equity_curve,
            initial_equity=initial_equity,
        )
        return {
            "metrics": metrics,
            "trade_log": trade_log,
            "equity_curve": equity_curve,
        }

    def _compute_metrics(
        self,
        trade_log: pd.DataFrame,
        equity_curve: pd.DataFrame,
        initial_equity: float,
    ) -> Dict[str, float]:
        """Compute summary metrics from trade log and equity curve."""

        if trade_log.empty:
            return BacktestResult(
                equity=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                total_return=0.0,
                num_trades=0,
            ).as_dict()

        wins = trade_log[trade_log["pnl"] > 0]
        losses = trade_log[trade_log["pnl"] <= 0]
        equity = trade_log["pnl"].sum()
        win_rate = len(wins) / len(trade_log)
        profit_factor = wins["pnl"].sum() / abs(losses["pnl"].sum()) if not losses.empty else float("inf")
        equity_series = equity_curve["equity"]
        peak = equity_series.cummax()
        drawdowns = peak - equity_series
        max_drawdown = drawdowns.max()
        ending_equity = equity_series.iloc[-1]
        total_return = ending_equity - initial_equity
        total_return_pct = total_return / initial_equity if initial_equity else 0.0
        gross_profit = wins["pnl"].sum()
        gross_loss = losses["pnl"].sum()

        return BacktestResult(
            equity=equity,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            total_return=total_return,
            num_trades=len(trade_log),
        ).as_dict() | {
            "initial_equity": initial_equity,
            "ending_equity": ending_equity,
            "total_return_pct": total_return_pct,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
        }
