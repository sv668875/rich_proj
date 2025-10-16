# 高勝率加密貨幣日內交易策略與 Python 程式交易系統藍圖  
# High Win-Rate Crypto Intraday Strategy & Python Trading System Blueprint

本專案提供以 **虛擬環境** 為基礎的模組化量化交易系統範例，涵蓋策略、回測、實盤執行、介面與資料管理模組，並示範兩套高期望值策略（ADX 趨勢動能、RSI 動能突破）。  
This repository delivers a virtual-environment-first, modular quantitative trading system skeleton, featuring strategy, backtest, execution, UI, and storage modules plus two high-expected-value strategies (ADX Trend Momentum, RSI Momentum Breakout).

---

## 1. 建置流程 | Setup Workflow

1. 建立虛擬環境：  
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```  
   Create a virtual environment and install dependencies.

2. 快速語法檢查：  
   ```bash
   python -m compileall src
   ```  
   Run a quick syntax check.

3. 單元測試（如有擴充）：  
   ```bash
   pytest
   ```  
   Execute unit tests when available.

4. CLI 回測：  
   ```bash
   python app.py --backtest --days 30 --strategy adx_trend --capital 10000
   ```  
   Run command-line backtests with configurable strategy and initial capital.

---

## 2. 專案結構 | Project Structure

```
rich_proj/
├── app.py                          # CLI 進入點 | CLI entry point
├── config/
│   └── strategy.yaml               # 策略參數雙語設定 | Bilingual strategy config
├── data/
│   └── results/                    # 回測輸出 | Backtest outputs
├── requirements.txt                # 套件需求 | Dependencies
├── src/
│   └── trading_system/
│       ├── __init__.py
│       ├── backtest.py             # 回測協調器 | Backtest orchestrator
│       ├── config_loader.py        # 組態載入 | Config loader
│       ├── data_fetcher.py         # 歷史資料抓取 | Data fetcher
│       ├── strategies/
│       │   ├── base.py             # 策略基底 | Strategy base
│       │   ├── adx_trend.py        # ADX 策略 | ADX strategy
│       │   ├── rsi_momentum.py     # RSI 策略 | RSI strategy
│       │   └── registry.py         # 策略註冊表 | Strategy registry
│       ├── backtesting/
│       │   └── engine.py           # 回測引擎 | Backtesting engine
│       ├── execution/
│       │   └── live_executor.py    # 實盤執行藍本 | Live execution blueprint
│       ├── interfaces/
│       │   └── ui_dashboard.py     # Streamlit 儀表板 | Streamlit dashboard
│       └── storage/
│           └── repository.py       # 策略存取 | Strategy repository
└── tests/
    ├── test_config_loader.py       # 組態測試 | Config loader test
    ├── test_backtest_engine.py     # 回測引擎測試 | Backtest engine test
    └── test_strategies.py          # 策略測試 | Strategy test
```

---

## 3. 核心模組說明 | Core Module Overview

- **策略模組 | Strategy Module**  
  `src/trading_system/strategies/` 提供 `Strategy` 抽象類別與高勝率策略實作，所有判斷條件皆以中文註解清楚標註。

- **回測系統 | Backtesting System**  
  `src/trading_system/backtest.py` 與 `backtesting/engine.py` 連結資料、策略與績效計算，示範如何輸出勝率、利潤因子等指標。

- **自動交易藍圖 | Live Execution Blueprint**  
  `execution/live_executor.py` 描述連接交易所 API 的流程，含中文風控註記，可依需求對接 CCXT 或 python-binance。

- **資料取得 | Data Acquisition**  
  `data_fetcher.py` 內建 Binance 公開 REST API 取價功能（`data_source: binance`），若 API 無回應會退回示意資料，方便離線開發。

- **使用者介面 | User Interface**  
  `ui_app.py` 搭配 `interfaces/ui_dashboard.py`，透過 Streamlit 提供策略選擇、參數調整、回測觸發與結果視覺化，全程雙語提示。

- **策略儲存管理 | Strategy Storage**  
  `storage/repository.py` 使用 JSON/CSV 快速儲存策略參數與紀錄，可延伸至 SQLite。

---

## 4. 高勝率策略摘要 | Strategy Highlights

1. **ADX 趨勢動能策略 | ADX Trend Momentum**  
   - EMA200 過濾主趨勢、Stochastic RSI 偵測拉回、ADX > 50 確認動能。  
   - 目標：在強勢趨勢中以 1:1 盈虧比捕捉高勝率拉回進場點。

2. **RSI 動能突破策略 | RSI Momentum Breakout**  
   - 5 日 RSI 突破 70 才進場，維持動能直到 RSI 回落。  
   - 目標：減少持倉時間、凸顯趨勢行情，回避盤整風險。

所有策略註解以「中文 | English」格式說明指標與風控邏輯，方便日後擴寫。

---

## 5. 下一步建議 | Suggested Next Steps

1. **啟用真實行情 | Enable Live Market Feed**：將 `config/strategy.yaml` 的 `data_source` 設為 `binance` 並設定交易對，即可透過公開 API 取得真實 K 線。  
2. **完善回測分析 | Enhance Backtesting Metrics**：新增 Sharpe Ratio、最大回撤等分析器並撰寫 pytest 測試。  
3. **實盤接入 | Connect Live Trading**：串接 Binance/CCXT 並加入 API 金鑰管理與重試邏輯。  
4. **介面擴充 | Expand UI**：加入參數即時調整、策略切換與通知推播。  
5. **風控深化 | Deepen Risk Controls**：實作每日停損、部位上限與部位網格追蹤。

---

## 6. UI 操作指南 | UI Usage Guide

1. 啟動服務：  
   `streamlit run src/trading_system/ui_app.py`

2. 於左側選單設定：  
   - 選擇策略（ADX / RSI）  
   - 輸入交易對（例如 `ETHUSDT`）與時間框架  
   - 指定資料來源（Binance / 本地示意）與回測天數  
   - 輸入初始資本金額，調整單筆風險百分比與是否保存結果檔案

3. 點擊「執行回測」後，主畫面即顯示：  
   - 勝率、利潤因子、最大回撤等指標  
   - 初始資本、期末資本與預估獲益/損失（含百分比）  
   - 資產淨值曲線（Equity Curve）  
   - 交易紀錄表格（進出場價、方向、PnL）

4. 切換「AI 訓練」分頁，可設定訓練迭代次數、資本與風控，並即時檢視 AI 模擬搜尋各組參數的回測結果。

全數操作皆提供繁中｜英文提示，無需命令列亦可完成策略回測流程。  
All interactions are bilingual, allowing you to conduct backtests entirely from the UI without touching the CLI.

---

如需擴充請遵循雙語註解規範與虛擬環境作業流程，確保系統維持模組化與高可讀性。  
Follow the bilingual annotation standard and virtual-environment workflow when extending the system to keep it modular and maintainable.
