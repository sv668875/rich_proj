# Repository Guidelines

## Project Structure & Module Organization
- `src/trading_system/` holds the core ETH delta-neutral modules (`backtest.py`, `data_fetcher.py`, `ui_app.py`, etc.).
- `app.py` is the CLI entry point for scripted backtests.
- `config/strategy.yaml` stores strategy parameters; update via Streamlit sidebar or CLI.
- `data/` is reserved for cached outputs (e.g., `results/backtest_summary.json`).
- `.venv/` is ignored; create your own virtual environment locally.

## Build, Test, and Development Commands
- Create venv and install deps:
  ```bash
  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
  ```
- Launch UI workstation:
  ```bash
  streamlit run src/trading_system/ui_app.py
  ```
- Run CLI backtest over last 30 days:
  ```bash
  python app.py --backtest --days 30
  ```
- Quick syntax check:
  ```bash
  python -m compileall src
  ```

## Coding Style & Naming Conventions
- Use Python 3.9+ with `black`-compatible formatting (4-space indent, double quotes).
- Prefer dataclasses for structured data; keep modules cohesive (config, analytics, risk).
- Name modules in lowercase with underscores; classes in PascalCase; functions and variables in snake_case.
- Keep UI strings concise and actionable; avoid long-form prose in code.

## Localization & Communication
- 回覆使用者請以使用者語言（預設繁體中文）為主，如需補充可附英文並保持專業口吻。
- 所有對外顯示的文案、UI 標籤與文件需提供繁體中文與英文對照，例如 `st.caption("即時資金費率 | Funding Rate")`。
- YAML、README、配置樣板中的關鍵段落亦請維持雙語，以利內外部協作。

## Testing Guidelines
- Unit tests should live under `tests/` (create if absent) mirroring package paths.
- Use `pytest` for new tests; name files `test_<module>.py` and functions `test_<behavior>()`.
- When adding backtest logic, include regression fixtures covering funding series edge cases.
- For UI changes, add smoke checks that validate critical callbacks if possible.

## Commit & Pull Request Guidelines
- Write commits in imperative mood (e.g., “Add funding Monte Carlo helper”).
- Keep commits small and scoped: one feature or fix per commit.
- Pull requests must describe strategy impact, include run commands (`streamlit` or CLI), and note any config or dependency updates.
- Link to related issues or tickets; attach screenshots/GIFs for UI changes when practical.

## Security & Configuration Tips
- Do not commit API keys; use environment variables or `.env` files ignored by git.
- Rate limits exist on Binance endpoints—add retries/backoff rather than tight loops.
- Validate external responses before using them in risk calculations to avoid NaNs propagating through analytics.
