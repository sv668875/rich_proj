"""Persistence utilities.

策略儲存管理：負責配置、交易記錄與績效的持久化 | Handle config and trade log persistence.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


@dataclass
class StrategyProfile:
    """策略檔案結構 | Strategy profile definition."""

    name: str
    description: str
    parameters: Dict[str, Any]


class StrategyRepository:
    """Simple JSON/CSV storage for strategy artifacts.

    策略資料庫：以檔案方式儲存策略設定與回測數據 | Persist strategy configs and logs to disk.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_profile(self, profile: StrategyProfile) -> Path:
        path = self.base_dir / f"{profile.name}.json"
        pd.Series(profile.parameters).to_json(path, force_ascii=False)
        return path

    def list_profiles(self) -> List[Path]:
        return sorted(self.base_dir.glob("*.json"))
