"""RSI momentum breakout strategy implementation.

RSI 動能突破策略：當短期 RSI 強勢突破時追隨趨勢，確保高期望值 | Momentum breakout pattern.
"""

import pandas as pd

from .base import Strategy


class RSIMomentumStrategy(Strategy):
    """追勢型策略 | Trend-following momentum strategy."""

    def generate_signals(self, candles: pd.DataFrame) -> pd.DataFrame:
        candles = candles.copy()
        candles["timestamp"] = pd.to_datetime(candles["timestamp"])
        candles.set_index("timestamp", inplace=True)

        rsi_period = int(self.parameters.get("rsi_period", 5))
        rsi_entry_threshold = float(self.parameters.get("rsi_entry_threshold", 70))
        trailing_exit_threshold = float(self.parameters.get("rsi_exit_threshold", 70))

        candles["rsi"] = self._compute_rsi(candles["close"], period=rsi_period)

        candles["signal"] = ""
        candles.loc[candles["rsi"] > rsi_entry_threshold, "signal"] = "buy"
        candles.loc[candles["rsi"] < (100 - rsi_entry_threshold), "signal"] = "sell"

        candles["target_profit"] = candles["close"].pct_change().fillna(0)
        candles["exit_rsi_level"] = trailing_exit_threshold

        return candles.reset_index()

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI.

        透過平滑平均計算 RSI，供策略判定動能 | Calculate RSI for momentum filtering.
        """

        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        rsi = rsi.where(avg_loss != 0, 100)
        rsi = rsi.where(avg_gain != 0, 0)
        flat_mask = (avg_gain == 0) & (avg_loss == 0)
        rsi = rsi.where(~flat_mask, 50)

        return rsi.fillna(50)
