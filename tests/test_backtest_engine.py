from pathlib import Path

import pandas as pd

from src.trading_system.backtesting.engine import BacktestEngine
from src.trading_system.config_loader import (
    StrategyConfig,
    MetadataConfig,
    MarketConfig,
    RiskConfig,
    DataConfig,
)
from src.trading_system.strategies.base import Strategy


class DummyStrategy(Strategy):
    def generate_signals(self, candles: pd.DataFrame) -> pd.DataFrame:
        candles = candles.copy()
        candles["signal"] = ""
        candles.loc[candles.index[::2], "signal"] = "buy"
        candles["target_profit"] = 10.0
        return candles


def _config() -> StrategyConfig:
    return StrategyConfig(
        metadata=MetadataConfig(name="dummy", symbol="ETHUSDT", notes="測試 | Test"),
        market=MarketConfig(interval="1h", base_currency="ETH", quote_currency="USDT"),
        risk=RiskConfig(risk_per_trade=1.0, max_daily_loss=5.0, slippage=0.1),
        data=DataConfig(data_source="local", output_dir=Path("data/results")),
        parameters={},
    )


def test_backtest_engine_produces_metrics():
    candles = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="h"),
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104, 105],
            "volume": [1000] * 6,
        }
    )

    engine = BacktestEngine(strategy_cls=DummyStrategy, config=_config())
    results = engine.run(candles=candles)
    metrics = results["metrics"]
    trade_log = results["trade_log"]

    assert metrics["num_trades"] > 0
    assert not trade_log.empty
    assert 0 <= metrics["win_rate"] <= 1
    assert metrics["initial_equity"] == 1.0
