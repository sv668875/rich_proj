from pathlib import Path

import pandas as pd

from src.trading_system.config_loader import StrategyConfig, MetadataConfig, MarketConfig, RiskConfig, DataConfig
from src.trading_system.strategies.rsi_momentum import RSIMomentumStrategy


def _build_config(parameters: dict) -> StrategyConfig:
    return StrategyConfig(
        metadata=MetadataConfig(name="rsi_momentum", symbol="ETHUSDT", notes="測試 | Test"),
        market=MarketConfig(interval="1h", base_currency="ETH", quote_currency="USDT"),
        risk=RiskConfig(risk_per_trade=1.0, max_daily_loss=5.0, slippage=0.1),
        data=DataConfig(data_source="local", output_dir=Path("data/results")),
        parameters=parameters,
    )


def test_rsi_momentum_generates_buy_signal():
    config = _build_config({"rsi_period": 3, "rsi_entry_threshold": 60, "rsi_exit_threshold": 55})
    strategy = RSIMomentumStrategy(config=config)
    timestamps = pd.date_range("2024-01-01", periods=20, freq="h")
    close_prices = pd.Series(range(100, 120))
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close_prices - 1,
            "high": close_prices + 2,
            "low": close_prices - 2,
            "close": close_prices,
            "volume": 1000,
        }
    )

    signals = strategy.generate_signals(df)

    assert "buy" in signals["signal"].values, "RSI 動能策略應在強勢趨勢中產生買進訊號 | RSI strategy should emit buy signal"
