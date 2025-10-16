"""Strategy base interface.

策略基底介面：定義共用方法與模擬流程 | Base class for trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from ..config_loader import StrategyConfig


@dataclass
class TradeRecord:
    """單筆交易紀錄 | Single trade entry."""

    timestamp: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    pnl: float


class Strategy(ABC):
    """Abstract strategy template.

    抽象策略模板：定義計算流程與模擬方法 | Define pipeline for generating trading signals.
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.parameters = config.parameters

    @abstractmethod
    def generate_signals(self, candles: pd.DataFrame) -> pd.DataFrame:
        """Compute signal columns from OHLCV."""

    def simulate(
        self,
        candles: pd.DataFrame,
        initial_equity: float = 1.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run simplified position simulation.

        根據信號執行部位控管並回傳交易紀錄與資金曲線 | Derive trade log and equity curve from signals.
        """

        signals = self.generate_signals(candles).reset_index(drop=True)
        if signals.empty:
            empty_log = pd.DataFrame(columns=TradeRecord.__annotations__.keys())
            equity_curve = pd.DataFrame(
                [{"timestamp": pd.Timestamp.utcnow(), "equity": initial_equity}]
            )
            return empty_log, equity_curve

        risk_pct = float(self.config.risk.risk_per_trade) / 100 if self.config.risk.risk_per_trade else 0.01
        slippage_pct = float(self.config.risk.slippage) / 100 if self.config.risk.slippage else 0.0

        equity = float(initial_equity)
        trades: List[TradeRecord] = []
        equity_points: List[Dict[str, object]] = []

        for idx in range(len(signals) - 1):
            row = signals.iloc[idx]
            next_row = signals.iloc[idx + 1]
            equity_points.append({"timestamp": row["timestamp"], "equity": equity})

            signal = row.get("signal", "")
            if signal not in {"buy", "sell"}:
                continue

            entry_price = float(row["close"])
            exit_price = float(next_row["close"])
            price_change_pct = (exit_price - entry_price) / entry_price
            if signal == "sell":
                price_change_pct *= -1

            price_change_pct -= slippage_pct
            trade_capital = equity * risk_pct
            pnl_value = trade_capital * price_change_pct
            equity += pnl_value

            trades.append(
                TradeRecord(
                    timestamp=row["timestamp"],
                    direction="long" if signal == "buy" else "short",
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=pnl_value,
                )
            )

        last_timestamp = signals.iloc[-1]["timestamp"]
        equity_points.append({"timestamp": last_timestamp, "equity": equity})

        trade_log = pd.DataFrame([trade.__dict__ for trade in trades])
        equity_curve = pd.DataFrame(equity_points)
        return trade_log, equity_curve

    def indicator_context(self) -> Dict[str, float]:
        """Return user-facing indicator descriptions.

        可提供 UI 顯示的指標描述 | Provide context for UI dashboards.
        """

        return {}
