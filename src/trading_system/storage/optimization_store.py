"""Utilities for persisting optimisation outputs.

最佳化結果儲存工具：負責將回測績效表保存於資料夾並提供查詢 | Persist optimisation tables for later comparison.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import json
import pandas as pd

from ..optimization.engine import OptimisationMetadata


@dataclass(frozen=True)
class PersistedRun:
    """Record describing a stored optimisation result."""

    csv_path: Path
    metadata_path: Path
    metadata: OptimisationMetadata


class OptimisationStore:
    """Persist optimisation DataFrames and metadata."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path("data/optimization")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        results: pd.DataFrame,
        metadata: OptimisationMetadata,
    ) -> PersistedRun:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        folder = self.base_dir / metadata.strategy_name
        folder.mkdir(parents=True, exist_ok=True)

        stem = f"{metadata.method}_{timestamp}"
        csv_path = folder / f"{stem}.csv"
        meta_path = folder / f"{stem}.json"

        results.to_csv(csv_path, index=False, encoding="utf-8")
        metadata_payload = asdict(metadata)
        metadata_payload["started_at"] = metadata.started_at.isoformat()
        meta_payload = json.dumps(metadata_payload, ensure_ascii=False, indent=2)
        meta_path.write_text(meta_payload, encoding="utf-8")

        return PersistedRun(csv_path=csv_path, metadata_path=meta_path, metadata=metadata)

    def load_history(
        self,
        strategy_name: Optional[str] = None,
        method: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Aggregate historical optimisation runs."""

        folders: List[Path]
        if strategy_name:
            folder = self.base_dir / strategy_name
            if not folder.exists():
                return pd.DataFrame()
            folders = [folder]
        else:
            folders = [p for p in self.base_dir.iterdir() if p.is_dir()]

        frames: List[pd.DataFrame] = []
        for folder in folders:
            csv_files = sorted(folder.glob("*.csv"), reverse=True)
            if limit:
                csv_files = csv_files[:limit]
            for csv_path in csv_files:
                df = pd.read_csv(csv_path)
                df["source_file"] = csv_path.name
                if method:
                    methods = df.get("method")
                    if methods is not None and not methods.str.contains(method).any():
                        continue
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        combined.sort_values(["total_return_pct", "win_rate"], ascending=[False, False], inplace=True)
        combined.reset_index(drop=True, inplace=True)
        return combined
