"""Backtest orchestration helpers.

歷史回測總控模組：串接資料、策略與績效分析 | Coordinate data, strategy, and analytics for backtests.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Type

import pandas as pd

from .backtesting.engine import BacktestEngine
from .config_loader import StrategyConfig
from .data_fetcher import HistoricalDataFetcher
from .strategies.base import Strategy
from .strategies.registry import STRATEGY_REGISTRY


def run_backtest(
    strategy_name: str,
    config: StrategyConfig,
    lookback_days: int,
    persist: bool = True,
    initial_equity: float = 1.0,
):
    """Load strategy and execute backtest over requested window.

    執行回測：讀取策略組態、載入歷史資料、輸出績效摘要 | Run backtest pipeline end-to-end.
    """

    strategy_cls = _resolve_strategy(strategy_name)
    fetcher = HistoricalDataFetcher(
        symbol=config.metadata.symbol,
        interval=config.market.interval,
        source=config.data.data_source,
    )

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)
    candles = fetcher.fetch_ohlcv(start=start_time, end=end_time)

    engine = BacktestEngine(
        strategy_cls=strategy_cls,
        config=config,
    )
    results = engine.run(candles=candles, initial_equity=initial_equity)
    if persist:
        _persist_summary(results=results, output_dir=config.data.output_dir)
    return results


def _resolve_strategy(strategy_name: str) -> Type[Strategy]:
    """Return strategy class from registry.

    透過策略註冊表取得對應策略類別，若不存在則拋出錯誤 | Lookup strategy class by name.
    """

    try:
        return STRATEGY_REGISTRY[strategy_name]
    except KeyError as exc:
        raise ValueError(
            f"策略 {strategy_name} 未註冊，請確認 config 或策略模組 | Strategy not registered"
        ) from exc


def _persist_summary(results: Dict[str, float], output_dir: Path) -> None:
    """Save backtest metrics as JSON & CSV.

    將回測績效摘要落地，便於後續 UI 與分析使用 | Store performance stats for later inspection.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "backtest_summary.json"
    equity_path = output_dir / "backtest_equity_curve.csv"
    trades_path = output_dir / "backtest_trades.csv"

    metrics = results.get("metrics", {})
    trade_log = results.get("trade_log", pd.DataFrame())
    equity_curve = results.get("equity_curve", pd.DataFrame())

    pd.DataFrame([metrics]).to_json(json_path, orient="records", force_ascii=False)
    if not equity_curve.empty:
        equity_curve.to_csv(equity_path, index=False)
    if not trade_log.empty:
        trade_log.to_csv(trades_path, index=False)
