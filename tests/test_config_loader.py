import tempfile
from pathlib import Path

import yaml

from src.trading_system.config_loader import load_strategy_config


def test_load_strategy_config_roundtrip(tmp_path: Path) -> None:
    data = {
        "metadata": {"name": "demo", "symbol": "BTCUSDT", "notes": "測試 | Test"},
        "market": {
            "interval": "1h",
            "base_currency": "BTC",
            "quote_currency": "USDT",
        },
        "risk": {"risk_per_trade": 1.0, "max_daily_loss": 5.0, "slippage": 0.1},
        "data": {"data_source": "local", "output_dir": str(tmp_path / "results")},
        "parameters": {"ema_period": 200},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    config = load_strategy_config(cfg_path)

    assert config.metadata.name == "demo"
    assert config.market.interval == "1h"
    assert config.data.output_dir.name == "results"
