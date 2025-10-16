#!/usr/bin/env python3
"""CLI for running crypto strategy backtests and live trading.

終端機介面執行高勝率加密貨幣日內交易策略 | CLI entry point for high win-rate crypto intraday strategies
"""

import argparse
from pathlib import Path

from src.trading_system import backtest
from src.trading_system.config_loader import load_strategy_config


def parse_args() -> argparse.Namespace:
    """Create CLI argument parser.

    建立命令列參數，支援回測與實盤模擬 | Build CLI options for backtest/live modes.
    """

    parser = argparse.ArgumentParser(
        description="高勝率策略回測與執行 | High win-rate strategy backtest runner"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/strategy.yaml"),
        help="策略設定檔路徑 | Strategy config path",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="adx_trend",
        help="選擇策略名稱 | Choose strategy name",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="執行歷史回測 | Run historical backtest",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="回測天數 (僅限回測模式) | Number of days for backtest",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="初始資本 (美元) | Initial capital in USD",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="啟動實盤模擬 | Start live simulation",
    )
    return parser.parse_args()


def main() -> None:
    """Entry function for CLI.

    根據命令列選項讀取配置並執行對應流程 | Dispatch tasks based on CLI options.
    """

    args = parse_args()
    config = load_strategy_config(args.config)

    if args.backtest:
        results = backtest.run_backtest(
            strategy_name=args.strategy,
            config=config,
            lookback_days=args.days,
            initial_equity=args.capital,
        )
        metrics = results.get("metrics", {})
        print("回測完成 | Backtest Summary")
        for key, value in metrics.items():
            print(f"{key}: {value}")
    elif args.live:
        raise NotImplementedError(
            "實盤模擬模組尚在藍圖階段 | Live trading loop not yet implemented"
        )
    else:
        raise SystemExit(
            "請使用 --backtest 或 --live 選項 | Use --backtest or --live switches"
        )


if __name__ == "__main__":
    main()
