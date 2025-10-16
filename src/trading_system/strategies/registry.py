"""Strategy registry.

策略註冊表：集中管理策略名稱對應的類別 | Map strategy identifiers to classes.
"""

from typing import Dict, Type

from .adx_trend import ADXTrendStrategy
from .rsi_momentum import RSIMomentumStrategy
from .base import Strategy

STRATEGY_REGISTRY: Dict[str, Type[Strategy]] = {
    "adx_trend": ADXTrendStrategy,
    "rsi_momentum": RSIMomentumStrategy,
}
