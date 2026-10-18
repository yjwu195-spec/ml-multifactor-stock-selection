# Packaging release - 2026-10-19

- Created an independent folder without changing the original research directory.
- Kept source data acquisition and computation; did not substitute a results viewer for the backtest.
- Made improved-v2 paths repository-relative and added `run_research.py`.
- Replaced direct token configuration with environment-only `TUSHARE_TOKEN`, optional local `.env`, and a blank example.
- Declared previously missing PyArrow, XGBoost, LightGBM, dotenv and tabulate dependencies.
- Added English/Chinese README, data/factor/method/result documentation and explicit research limitations.
- Included historical aggregate CSV/JSON, three figures, prior Word/PPT and an HTML overview.
- Excluded raw/processed market data, full scored panels, pretrained models, temporary files and old backtest-Sharpe search code with external-directory dependencies.
- Added syntax, path, credential and result-regression checks. Verification details are recorded separately.
- No remote creation, commit, push, new data download or full retraining was performed.

Review approach references the local `paper-replicator` skill (existing-project data-flow, credential, provenance and verification guidance). The requested scope is local packaging; no scheduled/remote automation was added.
