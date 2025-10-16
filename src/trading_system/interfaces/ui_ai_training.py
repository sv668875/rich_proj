"""AI training and optimisation dashboard.

AI 訓練與最佳化頁面：提供網格搜尋與貝式最佳化，並支援成果持久化與風控篩選 | Drive real optimisation
workflows with persistence and risk filters.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import yaml

from ..config_loader import StrategyConfig
from ..optimization import ParameterSpace, StrategyOptimizer, get_default_space
from ..storage.optimization_store import OptimisationStore


METHOD_DISPLAY_LABELS = {
    "grid": "網格搜尋 | Grid Search",
    "bayesian": "高斯過程 | GP Bayesian",
    "optuna": "Optuna TPE | Optuna TPE",
    "rl": "強化學習探索 | Reinforcement Learning",
}


def render_ai_training(base_config: StrategyConfig, strategy_options: List[str]) -> None:
    """Render AI optimisation tab with filtering and persistence."""

    if "ai_training_results" not in st.session_state:
        st.session_state["ai_training_results"] = pd.DataFrame()
    if "ai_training_metadata" not in st.session_state:
        st.session_state["ai_training_metadata"] = {}
    if "ai_training_method_options" not in st.session_state:
        st.session_state["ai_training_method_options"] = {}

    st.header("策略最佳化 | Strategy Optimisation")
    st.caption(
        "使用者友善的最佳化工作站：選擇方法、設定迭代次數，立即比較結果並保存歷史。 | "
        "A guided optimisation workspace: pick a method, run iterations, and compare/persist outcomes."
    )
    with st.expander("操作說明 | Quick How-To", expanded=False):
        st.markdown(
            "- 選擇策略、資料來源與回測參數，再挑選最佳化方法與迭代次數。\n"
            "- 按下 **開始最佳化 | Start Optimisation**，觀察即時進度及暫存結果。\n"
            "- 完成後會顯示最佳參數組合，並自動將結果存入 `data/optimization/` 方便長期比較。\n\n"
            "- Choose your strategy, data source, and backtest span, then select optimisation method/iterations.\n"
            "- Click **Start Optimisation** to watch live progress and interim scores.\n"
            "- When finished, the top parameter set is highlighted and saved under `data/optimization/` for review."
        )

    col_main, col_right = st.columns([3, 1])
    with col_main:
        selected_strategy = st.selectbox(
            "選擇訓練策略 | Select Training Strategy",
            options=strategy_options,
        )
        parameter_space = get_default_space(selected_strategy)
        st.caption("決定欲訓練的策略模組；不同策略擁有專屬參數空間。 | Pick which strategy to optimise; each exposes its own parameters.")
        symbol = st.text_input(
            "交易對 | Symbol",
            value=base_config.metadata.symbol,
            key="ai_symbol",
        )
        st.caption("指定回測/訓練用的交易對，影響資料抓取與回測結果。 | Choose the market symbol to backtest against; drives fetched data.")
        interval = st.selectbox(
            "時間框架 | Interval",
            options=["1m", "5m", "15m", "1h", "4h", "1d"],
            index=["1m", "5m", "15m", "1h", "4h", "1d"].index(base_config.market.interval)
            if base_config.market.interval in ["1m", "5m", "15m", "1h", "4h", "1d"]
            else 3,
            key="ai_interval",
        )
        st.caption("設定 K 線頻率；越短越貼近即時，越長越平滑但變化少。 | Controls candle timeframe; shorter intervals are more reactive, longer ones smoother.")
        data_source_options = ["binance", "binance_testnet", "local_csv"]
        data_source = st.selectbox(
            "資料來源 | Data Source",
            options=data_source_options,
            index=data_source_options.index(base_config.data.data_source)
            if base_config.data.data_source in data_source_options
            else 0,
            key="ai_data_source",
        )
        st.caption("選擇資料來源；Binance/測試網可取得即時市場，local_csv 用於離線樣本。 | Select data source; Binance/Testnet gives live markets, local_csv for offline samples.")
        lookback_days = st.slider(
            "回測天數 | Backtest Days",
            min_value=14,
            max_value=365,
            value=90,
            step=7,
            key="ai_lookback",
        )
        st.caption("定義歷史資料窗口，越長代表模型看到更多資料但耗時更久。 | How many historical days to backtest; longer windows offer context but take longer.")
        initial_capital = st.number_input(
            "初始資本 (USD) | Initial Capital (USD)",
            min_value=100.0,
            max_value=1_000_000.0,
            value=20_000.0,
            step=500.0,
            key="ai_capital",
        )
        st.caption("設定模擬起始資金，影響回測損益與資金曲線。 | Starting equity for simulations; impacts PnL and equity curve scale.")
        optimisation_method = st.selectbox(
            "最佳化方法 | Optimisation Method",
            options=list(METHOD_DISPLAY_LABELS.keys()),
            format_func=lambda x: METHOD_DISPLAY_LABELS.get(x, x),
            key="ai_method",
        )
        st.caption("選擇搜尋策略：網格、貝式、Optuna 或強化學習，會改變探索路徑。 | Choose optimisation approach; each drives different exploration behaviour.")
        default_iterations = 50 if optimisation_method in {"optuna", "rl"} else (25 if optimisation_method == "bayesian" else 40)
        iterations = st.slider(
            "迭代次數 / 評估筆數 | Iterations / Evaluations",
            min_value=5,
            max_value=300,
            value=default_iterations,
            step=5,
            key="ai_iterations",
        )
        st.caption("評估次數越多搜尋越全面，但需更多時間。 | More iterations improve coverage at the cost of runtime.")
        random_seed = st.number_input(
            "隨機種子 | Random Seed",
            min_value=0,
            max_value=10_000,
            value=42,
            step=1,
            key="ai_random_seed",
        )
        st.caption("固定隨機種子可重現結果；變換以增加探索多樣性。 | Fix seed for reproducibility; vary to diversify search paths.")

        method_options: Dict[str, Any] = {}
        with st.expander("進階設定 | Advanced Settings", expanded=False):
            if optimisation_method == "optuna":
                sampler_choice = st.selectbox(
                    "取樣器 | Sampler",
                    options=["tpe", "cmaes", "random"],
                    format_func=lambda x: {
                        "tpe": "TPE (預設) | TPE (Default)",
                        "cmaes": "CMA-ES | CMA-ES",
                        "random": "隨機探索 | Random Search",
                    }.get(x, x),
                    key="adv_optuna_sampler",
                )
                st.caption("控制 Optuna 探索方式：TPE 偏向貝式，CMA-ES 偏向全域搜尋。 | Pick Optuna sampler; TPE is Bayesian-inspired, CMA-ES explores globally.")
                method_options["optuna_sampler"] = sampler_choice
                if sampler_choice == "cmaes":
                    sigma0 = st.number_input(
                        "CMA-ES 初始步長 | CMA-ES Sigma0",
                        min_value=0.01,
                        max_value=5.0,
                        value=0.5,
                        step=0.05,
                        key="adv_optuna_sigma0",
                    )
                    st.caption("調整 CMA-ES 初始步長，越大代表初期探索範圍越廣。 | Larger sigma0 widens initial CMA-ES search radius.")
                    method_options["optuna_sigma0"] = float(sigma0)
                startup_trials = st.slider(
                    "啟動試數 | Startup Trials",
                    min_value=5,
                    max_value=100,
                    value=10,
                    step=5,
                    key="adv_optuna_startup",
                )
                st.caption("設定 TPE 前期純隨機抽樣次數，幫助建立統計分佈。 | Number of random warmup trials before TPE starts modelling.")
                method_options["optuna_startup_trials"] = int(startup_trials)
                pruner_choice = st.selectbox(
                    "提早停止器 | Pruner",
                    options=["none", "median"],
                    format_func=lambda x: "Median Pruner" if x == "median" else "停用 | None",
                    key="adv_optuna_pruner",
                )
                st.caption("Median Pruner 可早期淘汰表現差的 trial，節省計算時間。 | Median pruner stops underperforming trials to save time.")
                method_options["optuna_pruner"] = pruner_choice
                if pruner_choice == "median":
                    pruner_startup = st.slider(
                        "Pruner 啟動試數 | Pruner Startup Trials",
                        min_value=1,
                        max_value=50,
                        value=5,
                        step=1,
                        key="adv_optuna_pruner_startup",
                    )
                    st.caption("在多少 trial 後啟動 pruner；避免過早裁剪。 | Trials to run before pruner activates, preventing premature cuts.")
                    method_options["optuna_pruner_startup"] = int(pruner_startup)
                    pruner_warmup = st.slider(
                        "Pruner 暖身步數 | Pruner Warmup Steps",
                        min_value=0,
                        max_value=20,
                        value=0,
                        step=1,
                        key="adv_optuna_pruner_warmup",
                    )
                    st.caption("暖身步數越高，pruner 會更晚開始比較。 | Higher warmup delays pruning comparisons for stability.")
                    method_options["optuna_pruner_warmup"] = int(pruner_warmup)
                jobs = st.slider(
                    "平行工作數 | Parallel Jobs",
                    min_value=1,
                    max_value=4,
                    value=1,
                    step=1,
                    key="adv_optuna_jobs",
                )
                st.caption("允許 Optuna 並行評估 trial，需確保環境支援平行計算。 | Run multiple Optuna trials in parallel; requires parallel-safe environment.")
                method_options["optuna_jobs"] = int(jobs)
            elif optimisation_method == "rl":
                epsilon_start = st.slider(
                    "探索率初始值 | Initial Epsilon",
                    min_value=0.05,
                    max_value=1.0,
                    value=0.3,
                    step=0.05,
                    key="adv_rl_epsilon_start",
                )
                st.caption("起始探索率越高，初期越偏向嘗試新參數。 | Higher initial epsilon encourages early exploration.")
                epsilon_min = st.slider(
                    "探索率下限 | Minimum Epsilon",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.05,
                    step=0.01,
                    key="adv_rl_epsilon_min",
                )
                st.caption("下降後的最小探索率，避免過早陷入局部解。 | Floor for exploration rate to avoid premature convergence.")
                epsilon_decay = st.slider(
                    "探索率衰減 | Epsilon Decay",
                    min_value=0.5,
                    max_value=0.99,
                    value=0.95,
                    step=0.01,
                    key="adv_rl_epsilon_decay",
                )
                st.caption("控制探索率下降速度；越小代表轉向 exploitation 越快。 | Decay speed from exploration to exploitation.")
                reward_mix = st.slider(
                    "報酬/勝率權重 | Reward/Win Mix",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.05,
                    key="adv_rl_reward_mix",
                )
                st.caption("平衡總報酬與勝率：靠近 1 偏好報酬，靠近 0 偏好勝率。 | Trade-off between total return and win rate in reward shaping.")
                pool_multiplier = st.slider(
                    "候選池倍率 | Candidate Pool Multiplier",
                    min_value=1,
                    max_value=8,
                    value=4,
                    step=1,
                    key="adv_rl_pool_multiplier",
                )
                st.caption("控制隨機候選集合大小，越大探索更廣但耗時更多。 | Sets random candidate pool size; larger pools broaden exploration.")
                method_options.update(
                    {
                        "rl_epsilon_start": float(epsilon_start),
                        "rl_epsilon_min": float(epsilon_min),
                        "rl_epsilon_decay": float(epsilon_decay),
                        "rl_reward_mix": float(reward_mix),
                        "rl_pool_multiplier": int(pool_multiplier),
                    }
                )
            elif optimisation_method == "bayesian":
                st.caption("使用自適應高斯過程；可藉由調整迭代次數與隨機種子控制探索。 | Adaptive GP search; tune iterations & seed for exploration.")
            else:
                st.caption("網格搜尋會逐一評估全部組合，迭代次數會自動限制在組合總數。 | Grid search will iterate through the cartesian space.")

    with col_right:
        st.write("風控設定 | Risk Settings")
        risk_per_trade = st.slider(
            "單筆風險(%) | Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=float(base_config.risk.risk_per_trade),
            step=0.5,
            key="ai_risk_per_trade",
        )
        st.caption("調整每筆交易投入資金比例；越高回測波動越大。 | Controls capital per trade; higher risk amplifies swings.")
        slippage = st.slider(
            "滑點(%) | Slippage (%)",
            min_value=0.0,
            max_value=0.5,
            value=float(base_config.risk.slippage),
            step=0.05,
            key="ai_slippage",
        )
        st.caption("模擬成交滑價；越高代表市場流動性差或執行成本高。 | Simulated fill slippage to reflect execution costs.")
        st.write("績效篩選 | Performance Filters")
        min_win_rate = st.slider(
            "最低勝率 | Min Win Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            key="ai_min_win_rate",
        )
        st.caption("設定策略最低可接受勝率，協助篩掉過度冒險的組合。 | Minimum acceptable win rate to filter high-risk parameter sets.")
        max_drawdown_pct_raw = st.slider(
            "最大回撤 (百分比) | Max Drawdown (%)",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            key="ai_max_drawdown_pct",
        )
        st.caption("限制最大資金回撤；越低代表風控更保守。 | Caps maximum drawdown to enforce risk tolerance.")
        max_drawdown_pct = max_drawdown_pct_raw / 100.0
        apply_filters = st.checkbox(
            "啟用風控篩選 | Apply Filters",
            value=True,
            key="ai_apply_filters",
        )
        st.caption("開關勝率與回撤篩選器；關閉可完整檢視所有試算。 | Enable/disable risk filters to view full results.")

    start_button = st.button("開始最佳化 | Start Optimisation", use_container_width=True)
    clear_button = st.button("清除快取 | Clear Session Cache", use_container_width=True)

    if clear_button:
        st.session_state["ai_training_results"] = pd.DataFrame()
        st.session_state["ai_training_metadata"] = {}
        st.success("已清除本次工作階段的最佳化結果 | Session optimisation cache cleared.")

    store = OptimisationStore()

    if start_button:
        runtime_config = deepcopy(base_config)
        runtime_config.metadata.name = selected_strategy
        runtime_config.metadata.symbol = symbol.upper()
        runtime_config.market.interval = interval
        runtime_config.data.data_source = data_source
        runtime_config.risk.risk_per_trade = risk_per_trade
        runtime_config.risk.slippage = slippage

        progress_bar = st.progress(0.0)
        status_placeholder = st.empty()
        preview_placeholder = st.empty()
        interim_records: List[Dict[str, float]] = []

        grid_capacity = _estimate_grid_size(parameter_space)
        if optimisation_method == "grid":
            total_iterations = min(iterations, grid_capacity)
        else:
            total_iterations = iterations

        def _on_eval(step: int, params: Dict[str, float], metrics: Dict[str, float]) -> None:
            record = dict(metrics)
            record["iteration"] = step
            record["params"] = params
            if "max_drawdown_pct" not in record:
                initial_equity_local = float(record.get("initial_equity", initial_capital or 1.0))
                record["max_drawdown_pct"] = (
                    float(record.get("max_drawdown", 0.0)) / initial_equity_local if initial_equity_local else 0.0
                )
            interim_records.append(record)
            preview_df = pd.DataFrame(interim_records)
            preview_df = _normalise_columns(preview_df)
            filtered_preview = (
                _apply_filters(preview_df, min_win_rate, max_drawdown_pct) if apply_filters else preview_df
            )
            progress_bar.progress(min(step / total_iterations, 1.0))
            status_placeholder.info(
                f"第 {step}/{total_iterations} 次評估完成 | Iteration {step}/{total_iterations} finished. "
                f"目前最佳報酬% | Current best return %: "
                f"{filtered_preview['total_return_pct'].max()*100 if not filtered_preview.empty else 0.0:.2f}%"
            )
            preview_columns = [
                "iteration",
                "win_rate",
                "profit_factor",
                "total_return_pct",
                "max_drawdown_pct",
                "num_trades",
                "params",
            ]
            display_df = filtered_preview if not filtered_preview.empty else preview_df
            available_columns = [col for col in preview_columns if col in display_df.columns]
            preview_placeholder.dataframe(display_df[available_columns])

        try:
            optimiser = StrategyOptimizer(
                base_config=runtime_config,
                strategy_name=selected_strategy,
                lookback_days=lookback_days,
                initial_equity=initial_capital,
                parameter_space=parameter_space,
                on_evaluation=_on_eval,
                method_kwargs=method_options,
            )

            results_df, metadata = optimiser.optimise(
                method=optimisation_method,
                iterations=total_iterations,
                random_seed=int(random_seed),
            )
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"最佳化失敗：{exc}")
            progress_bar.empty()
            status_placeholder.empty()
            preview_placeholder.empty()
            return

        results_df = _normalise_columns(results_df)
        st.session_state["ai_training_results"] = results_df
        st.session_state["ai_training_metadata"] = {
            "strategy": metadata.strategy_name,
            "method": metadata.method,
            "started_at": metadata.started_at.isoformat(),
            "iterations": metadata.iterations,
            "options": metadata.options,
            "random_seed": metadata.random_seed,
        }
        st.session_state["ai_training_method_options"] = dict(method_options)
        persisted = store.save(results_df, metadata)
        st.success(
            f"最佳化完成並已儲存 | Optimisation completed and persisted to {persisted.csv_path.name}."
        )

    current_results = st.session_state.get("ai_training_results", pd.DataFrame())
    if not current_results.empty:
        current_results = _normalise_columns(current_results)
        if "method_options" in current_results.columns:
            current_results["method_options"] = current_results["method_options"].apply(_ensure_params_dict)
        if "params" in current_results.columns:
            current_results["params"] = current_results["params"].apply(_ensure_params_dict)
        filtered_current = (
            _apply_filters(current_results, min_win_rate, max_drawdown_pct) if apply_filters else current_results
        )
        st.subheader("本次最佳結果 | Current Best Runs")
        metadata_state = st.session_state.get("ai_training_metadata", {})
        if metadata_state:
            method_label = METHOD_DISPLAY_LABELS.get(
                metadata_state.get("method", ""),
                metadata_state.get("method", ""),
            )
            started_at = metadata_state.get("started_at")
            caption_parts = [f"最佳化方法 | Method: {method_label}"]
            if started_at:
                caption_parts.append(f"啟動時間 | Started: {started_at}")
            seed_value = metadata_state.get("random_seed")
            if seed_value is not None:
                caption_parts.append(f"隨機種子 | Seed: {seed_value}")
            st.caption(" ｜ ".join(caption_parts))
        best_row = filtered_current.iloc[0] if not filtered_current.empty else current_results.iloc[0]
        metric_cols = st.columns(3)
        metric_cols[0].metric(
            "最佳總報酬% | Best Total Return %",
            f"{best_row['total_return_pct'] * 100:.2f}%",
            delta=f"{best_row['win_rate'] * 100:.1f}% 勝率 | Win Rate",
        )
        metric_cols[1].metric(
            "最大回撤% | Max Drawdown %",
            f"{best_row['max_drawdown_pct'] * 100:.1f}%",
            delta=f"PF {best_row.get('profit_factor', 0.0):.2f}",
        )
        metric_cols[2].metric(
            "交易次數 | Trades",
            f"{int(best_row.get('num_trades', 0))}",
        )
        best_params = _ensure_params_dict(best_row["params"])
        st.markdown("**最佳參數設定 | Top Parameter Set**")
        yaml_snippet = yaml.safe_dump(
            {"strategy": selected_strategy, "parameters": best_params},
            allow_unicode=True,
            sort_keys=True,
        )
        st.code(yaml_snippet, language="yaml")
        st.download_button(
            label="下載最佳參數 YAML | Download Best Params YAML",
            data=yaml_snippet,
            file_name=f"{selected_strategy}_best_params.yaml",
            mime="text/yaml",
        )
        st.caption(
            "可直接複製或下載以上參數至策略設定或 CLI，快速比對不同配置。 | "
            "Copy or download the snippet to try this configuration in configs/CLI."
        )
        method_opts_display = st.session_state.get("ai_training_method_options", {})
        if method_opts_display:
            st.caption("最佳化設定 | Optimiser Options")
            st.json(method_opts_display)
        table_columns = [
            "iteration",
            "win_rate",
            "profit_factor",
            "total_return_pct",
            "max_drawdown_pct",
            "num_trades",
            "params",
            "method",
            "method_options",
        ]
        selected_table = filtered_current if not filtered_current.empty else current_results
        display_table = selected_table.copy()
        if "method_options" in display_table.columns:
            display_table["method_options"] = display_table["method_options"].apply(_ensure_params_dict)
        if "params" in display_table.columns:
            display_table["params"] = display_table["params"].apply(_ensure_params_dict)
        available_columns = [col for col in table_columns if col in display_table.columns]
        st.dataframe(display_table[available_columns])

    st.subheader("歷史紀錄 | Historical Runs")
    history_df = store.load_history(strategy_name=selected_strategy, limit=10)
    if history_df.empty:
        st.caption("尚無歷史最佳化紀錄 | No historical optimisation snapshots found.")
    else:
        history_df = _normalise_columns(history_df)
        if "method_options" in history_df.columns:
            history_df["method_options"] = history_df["method_options"].apply(_ensure_params_dict)
        if "params" in history_df.columns:
            history_df["params"] = history_df["params"].apply(_ensure_params_dict)
        filtered_history = (
            _apply_filters(history_df, min_win_rate, max_drawdown_pct) if apply_filters else history_df
        )
        history_columns = [
            "strategy_name",
            "method",
            "method_options",
            "started_at",
            "win_rate",
            "profit_factor",
            "total_return_pct",
            "max_drawdown_pct",
            "num_trades",
            "params",
            "source_file",
        ]
        available_columns = [col for col in history_columns if col in filtered_history.columns]
        st.dataframe(filtered_history[available_columns])


def _ensure_params_dict(raw_params: Any) -> Dict[str, Any]:
    """Best-effort conversion of stored params into a dictionary."""

    if isinstance(raw_params, dict):
        return raw_params
    if isinstance(raw_params, str):
        try:
            loaded = yaml.safe_load(raw_params)
            if isinstance(loaded, dict):
                return loaded
        except Exception:  # pragma: no cover - defensive parsing
            return {"raw": raw_params}
    return {"raw": raw_params}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent numeric columns exist."""

    df = df.copy()
    if "max_drawdown_pct" not in df.columns and "max_drawdown" in df.columns:
        initial_equity = df.get("initial_equity", pd.Series(1.0, index=df.index))
        df["max_drawdown_pct"] = df["max_drawdown"] / initial_equity.replace(0, pd.NA)
        df["max_drawdown_pct"] = df["max_drawdown_pct"].fillna(0.0)
    if "win_rate" in df.columns:
        df["win_rate"] = df["win_rate"].clip(0, 1)
    if "total_return_pct" in df.columns:
        df["total_return_pct"] = df["total_return_pct"].fillna(0.0)
    return df


def _apply_filters(df: pd.DataFrame, min_win_rate: float, max_drawdown_pct: float) -> pd.DataFrame:
    """Apply win-rate and drawdown thresholds."""

    if df.empty:
        return df
    filtered = df[
        (df["win_rate"] >= min_win_rate)
        & (df["max_drawdown_pct"] <= max_drawdown_pct)
    ].copy()
    if filtered.empty:
        return df.iloc[0:0]
    filtered.sort_values(["total_return_pct", "win_rate"], ascending=[False, False], inplace=True)
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def _estimate_grid_size(parameter_space: ParameterSpace) -> int:
    """Estimate number of grid combinations for progress display."""

    size = 1
    for spec in parameter_space.specs:
        if spec.grid:
            size *= len(spec.grid)
        elif spec.bounds:
            size *= 5
    return int(size)
