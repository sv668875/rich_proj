"""Configuration helpers for strategies.

策略配置載入工具：管理 YAML 參數與資料結構 | Load strategy configs from YAML.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class MetadataConfig:
    """策略基本資訊 | Strategy metadata."""

    name: str
    symbol: str
    notes: str


@dataclass
class MarketConfig:
    """市場設定 | Market configuration."""

    interval: str
    base_currency: str
    quote_currency: str


@dataclass
class RiskConfig:
    """風險控管設定 | Risk management configuration."""

    risk_per_trade: float
    max_daily_loss: float
    slippage: float


@dataclass
class DataConfig:
    """資料來源設定 | Data source configuration."""

    data_source: str
    output_dir: Path


@dataclass
class StrategyConfig:
    """集合所有配置段落 | Aggregate strategy configuration."""

    metadata: MetadataConfig
    market: MarketConfig
    risk: RiskConfig
    data: DataConfig
    parameters: Dict[str, Any]


def load_strategy_config(path: Path) -> StrategyConfig:
    """Parse YAML config file into dataclasses.

    將 YAML 解析成策略設定物件，方便模組間共用 | Deserialize YAML into typed config.
    """

    with path.open(encoding="utf-8") as fh:
        raw_config = yaml.safe_load(fh)

    metadata = MetadataConfig(**raw_config["metadata"])
    market = MarketConfig(**raw_config["market"])
    risk = RiskConfig(**raw_config["risk"])
    data = DataConfig(
        data_source=raw_config["data"]["data_source"],
        output_dir=Path(raw_config["data"]["output_dir"]),
    )
    parameters = raw_config["parameters"]

    return StrategyConfig(
        metadata=metadata,
        market=market,
        risk=risk,
        data=data,
        parameters=parameters,
    )
