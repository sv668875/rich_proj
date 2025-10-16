"""ADX trend momentum strategy implementation.

ADX 趨勢動能策略實作：結合 EMA200、隨機 RSI 與 ADX 判斷高勝率順勢進場 | Illustrative implementation.
"""

import pandas as pd

from .base import Strategy


class ADXTrendStrategy(Strategy):
    """高勝率趨勢回檔策略 | High win-rate pullback strategy."""

    def generate_signals(self, candles: pd.DataFrame) -> pd.DataFrame:
        candles = candles.copy()
        candles["timestamp"] = pd.to_datetime(candles["timestamp"])
        candles.set_index("timestamp", inplace=True)

        ema_window = int(self.parameters.get("ema_period", 200))
        adx_threshold = float(self.parameters.get("adx_threshold", 50))
        stochastic_period = int(self.parameters.get("stoch_period", 14))

        candles["ema_200"] = candles["close"].ewm(span=ema_window, adjust=False).mean()
        candles["high_high"] = candles["high"].rolling(window=stochastic_period).max()
        candles["low_low"] = candles["low"].rolling(window=stochastic_period).min()
        candles["stoch_k"] = (
            (candles["close"] - candles["low_low"]) / (candles["high_high"] - candles["low_low"])
        ) * 100
        candles["stoch_d"] = candles["stoch_k"].rolling(window=3).mean()

        candles["adx"] = self._approximate_adx(candles)

        candles["signal"] = ""
        buy_condition = (
            (candles["close"] > candles["ema_200"])
            & (candles["stoch_k"] < 20)
            & (candles["stoch_k"] > candles["stoch_d"])
            & (candles["adx"] > adx_threshold)
        )
        sell_condition = (
            (candles["close"] < candles["ema_200"])
            & (candles["stoch_k"] > 80)
            & (candles["stoch_k"] < candles["stoch_d"])
            & (candles["adx"] > adx_threshold)
        )
        candles.loc[buy_condition, "signal"] = "buy"
        candles.loc[sell_condition, "signal"] = "sell"

        risk_reward = float(self.parameters.get("risk_reward", 1.0))
        candles["target_profit"] = candles["close"].pct_change().fillna(0) * risk_reward

        return candles.reset_index()

    @staticmethod
    def _approximate_adx(candles: pd.DataFrame) -> pd.Series:
        """簡化版 ADX 計算，用於示意 | Approximate ADX without external libs."""

        high = candles["high"]
        low = candles["low"]
        close = candles["close"]

        plus_dm = (high.diff().clip(lower=0)).fillna(0)
        minus_dm = (-low.diff().clip(upper=0)).fillna(0)
        tr = pd.concat(
            [
                (high - low),
                (high - close.shift()).abs(),
                (low - close.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1).fillna(0)

        smooth = 14
        atr = tr.rolling(window=smooth).mean()
        plus_di = 100 * (plus_dm.rolling(window=smooth).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=smooth).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, pd.NA)) * 100
        return dx.rolling(window=smooth).mean().fillna(0)
