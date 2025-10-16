"""Streamlit application entry point.

高勝率策略監控介面入口：提供策略選擇、參數設定與回測結果視覺化 | Streamlit UI for strategy control and backtest visualization.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st

from src.trading_system.backtest import run_backtest
from src.trading_system.config_loader import load_strategy_config
from src.trading_system.interfaces.ui_dashboard import render_dashboard
from src.trading_system.interfaces.ui_ai_training import render_ai_training
from src.trading_system.strategies.registry import STRATEGY_REGISTRY


def main() -> None:
    """Bootstrap Streamlit dashboard with interactive controls."""

    st.set_page_config(page_title="高勝率策略儀表板 | High Win-Rate Dashboard", layout="wide")

    config = load_strategy_config(Path("config/strategy.yaml"))
    strategy_options = list(STRATEGY_REGISTRY.keys())

    if "backtest_results" not in st.session_state:
        st.session_state["backtest_results"] = None
    if "status_message" not in st.session_state:
        st.session_state["status_message"] = ""

    with st.sidebar:
        st.header("策略控制 | Strategy Controls")
        selected_strategy = st.selectbox(
            "選擇策略 | Select Strategy",
            options=strategy_options,
            index=strategy_options.index(config.metadata.name)
            if config.metadata.name in strategy_options
            else 0,
        )
        symbol = st.text_input("交易對 | Symbol", value=config.metadata.symbol)
        interval = st.selectbox(
            "時間框架 | Interval",
            options=["1m", "5m", "15m", "1h", "4h", "1d"],
            index=["1m", "5m", "15m", "1h", "4h", "1d"].index(config.market.interval)
            if config.market.interval in ["1m", "5m", "15m", "1h", "4h", "1d"]
            else 3,
        )
        data_source_options = ["binance", "binance_testnet", "local_csv"]
        data_source = st.selectbox(
            "資料來源 | Data Source",
            options=data_source_options,
            index=data_source_options.index(config.data.data_source)
            if config.data.data_source in data_source_options
            else 0,
        )
        lookback_days = st.slider(
            "回測天數 | Backtest Days",
            min_value=7,
            max_value=180,
            value=30,
            step=1,
        )
        initial_capital = st.number_input(
            "初始資本 (USD) | Initial Capital (USD)",
            min_value=100.0,
            max_value=1_000_000.0,
            value=10_000.0,
            step=100.0,
        )
        risk_per_trade = st.slider(
            "單筆風險(%) | Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=float(config.risk.risk_per_trade),
            step=0.5,
        )
        persist_results = st.checkbox("保存回測檔案 | Persist Results", value=True)
        run_button = st.button("執行回測 | Run Backtest", use_container_width=True)

    if run_button:
        runtime_config = deepcopy(config)
        runtime_config.metadata.symbol = symbol.upper()
        runtime_config.market.interval = interval
        runtime_config.data.data_source = data_source
        runtime_config.risk.risk_per_trade = risk_per_trade

        try:
            results = run_backtest(
                strategy_name=selected_strategy,
                config=runtime_config,
                lookback_days=lookback_days,
                persist=persist_results,
                initial_equity=initial_capital,
            )
            st.session_state["backtest_results"] = results
            st.session_state["status_message"] = "✅ 回測完成 | Backtest completed successfully."
        except Exception as exc:  # pylint: disable=broad-except
            st.session_state["backtest_results"] = None
            st.session_state["status_message"] = f"❌ 回測失敗：{exc}"

    results_state: Dict[str, object] | None = st.session_state.get("backtest_results")
    metrics = None
    equity_curve = None
    trade_log = None
    if results_state:
        metrics = results_state.get("metrics")
        equity_curve = results_state.get("equity_curve")
        trade_log = results_state.get("trade_log")

    tab_dashboard, tab_ai = st.tabs(["策略儀表板 | Strategy Dashboard", "AI 訓練 | AI Training"])

    with tab_dashboard:
        if st.session_state.get("status_message"):
            st.caption(st.session_state["status_message"])
        render_dashboard(metrics=metrics, equity_curve=equity_curve, trade_log=trade_log)

    with tab_ai:
        render_ai_training(base_config=config, strategy_options=strategy_options)


if __name__ == "__main__":
    main()
