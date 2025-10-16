"""Streamlit dashboard blueprint.

使用者介面藍圖：顯示回測結果、資金曲線與交易紀錄 | UI blueprint for showcasing backtest outcomes.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import streamlit as st


def render_dashboard(
    metrics: Optional[Dict[str, float]],
    equity_curve: Optional[pd.DataFrame],
    trade_log: Optional[pd.DataFrame],
) -> None:
    """Render dashboard components.

    顯示回測概要、資金曲線與交易紀錄 | Present backtest summary, equity curve, and trade log.
    """

    st.title("高勝率策略儀表板 | High Win-Rate Strategy Dashboard")

    if not metrics:
        st.info("請先於左側設定參數並執行回測 | Configure parameters on the left to run a backtest.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("勝率 | Win Rate", f"{metrics.get('win_rate', 0):.2%}")

    profit_factor = metrics.get("profit_factor", 0.0)
    profit_display = f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞"
    col2.metric("利潤因子 | Profit Factor", profit_display)

    col3.metric("最大回撤 | Max Drawdown", f"{metrics.get('max_drawdown', 0):.4f}")

    col4, col5, col6 = st.columns(3)
    total_return = metrics.get("total_return", 0.0)
    total_return_pct = metrics.get("total_return_pct", 0.0)
    col4.metric(
        "總報酬 | Total Return",
        f"{total_return:.2f}",
        delta=f"{total_return_pct * 100:.2f}%",
    )
    initial_equity = metrics.get("initial_equity", 0.0)
    ending_equity = metrics.get("ending_equity", initial_equity + total_return)
    col5.metric("初始資本 | Initial Capital", f"{initial_equity:.2f}")
    col6.metric("期末資本 | Ending Capital", f"{ending_equity:.2f}")

    st.metric("交易次數 | Trades", f"{metrics.get('num_trades', 0)}")
    st.caption(
        "預估獲益/損失 | Estimated P&L"
        f"：盈利 {metrics.get('gross_profit', 0.0):.2f}，虧損 {metrics.get('gross_loss', 0.0):.2f}"
    )

    st.subheader("資產淨值曲線 | Equity Curve")
    if equity_curve is not None and not equity_curve.empty:
        equity_curve_plot = equity_curve.copy()
        equity_curve_plot = equity_curve_plot.set_index("timestamp")
        st.line_chart(equity_curve_plot["equity"])
    else:
        st.write("尚未產生資金曲線資料 | No equity curve available.")

    st.subheader("交易紀錄 | Trade Log")
    if trade_log is not None and not trade_log.empty:
        st.dataframe(trade_log)
    else:
        st.write("尚無交易紀錄 | No trades executed.")
