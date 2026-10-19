# MH6805 Group 12 Machine Learning Stock Selection

[中文详细说明](README.zh-CN.md) | [Classmate guide 中文 English](docs/CLASSMATE_GUIDE.md) | [Data access and schema](docs/DATA_ACCESS.md) | [Methods](docs/METHODOLOGY.md) | [Results explained](docs/RESULTS_EXPLAINED.md)

**Course version, not the legacy internship backtest at the repository root.** This directory contains a separately implemented and executed course experiment comparing Logistic Regression, Random Forest, XGBoost and LightGBM on historical CSI 500 stocks. The older root-level research is retained as an archive; its figures must not be mixed with these results.

Authors: Wu Yangjie, Zhao Jingjie and Wang Zhaobo. Equal contributions.

## 1 Project goal

Can nonlinear combinations of financial and market factors improve stock ranking over a linear classifier, and do predictive gains translate into portfolio returns after trading costs? The objective is a transparent machine-learning comparison, not a claim of a production-ready trading strategy or a guaranteed return.

## 2 Start here without Python or paid data

1. Read [the classmate guide](docs/CLASSMATE_GUIDE.md) for the workflow and terminology.
2. Download and open [the Word report](deliverables/Group12_Final_Report.docx) or [the seven-page PDF](deliverables/Group12_Final_Report.pdf).
3. Download [the slides](deliverables/Group12_Presentation.pptx) and read [the speaker notes](deliverables/Group12_Speaker_Notes.md).
4. Download [the offline HTML report](deliverables/Group12_Results.html) and open it in a browser. GitHub's source preview does not run HTML.
5. Read [the results explanation](docs/RESULTS_EXPLAINED.md) before quoting any performance number.

All of those files and the aggregate CSV tables can be read without a Tushare token. Reading saved evidence is **not** rerunning training or the backtest.

## 3 Current status and what is public

The local real-data experiment completed data preparation, temporal model selection, fitting, predictive evaluation, portfolio ledgers, factor ablations, permutation importance, decile tests and cost/subperiod analysis. Nine deterministic unit tests and a full local computed-data audit passed. Saved evidence of that run is in `results/verification.json`; it is not proof that a fresh GitHub checkout has rerun the experiment.

This public package includes source, fixed configuration, tests, aggregate results, figures, English deliverables and editable LaTeX. It intentionally excludes licensed raw observations, stock-level processed panels/predictions, serialized models, a private proposal PDF and workstation metadata. A public clone alone can inspect results and run unit tests, but **cannot complete a fresh scientific run until authorized inputs are supplied**. There is no synthetic-data or hidden network fallback.

## 4 Research design

| Item | Fixed course design |
|---|---|
| Provider and raw window | Tushare cache, 2010-01-04 to 2025-04-30 |
| Universe | Historical CSI 500 snapshots dated no later than the signal; maximum age 62 calendar days |
| Factors | 25 features in 7 families; see methods for exact definitions |
| Signals and execution | Every 20 exchange sessions; signal at close, execution at next session open |
| Target | Next execution-open to following execution-open return minus CSI 500 return |
| Classes | Top 30% = 1, bottom 30% = 0; middle/unlabeled rows retained for inference |
| Primary portfolio | Top 10 scores, equal target value, long only |
| Cost | 10 bps per side on actual traded notional, with 0/5/10/20 bps sensitivity |
| Evaluation | Classification, ranking, daily portfolio outcomes, deciles, ablations, importance and time slices |

Financial availability uses the announcement date, not the reporting-period end. Indicators receive a conservative one-day lag; balance sheets use the later announcement/actual-announcement date plus one day. Reports older than 550 days become missing. Eligibility requires a current positive-volume quote, at least 60 stock observations, a computable 60-bar momentum and at most 40% missing features. The panel is intentionally unbalanced.

### Chronological samples

| Split | First signal | Last signal | Rebalance dates | Eligible rows |
|---|---|---|---:|---:|
| Train | 2010-05-05 | 2017-09-26 | 91 | 31,964 |
| Validation | 2017-10-31 | 2019-03-25 | 18 | 8,618 |
| Test | 2019-04-23 | 2020-09-11 | 18 | 8,985 |
| Later backtest | 2020-10-19 | 2025-04-01 | 55 | 27,467 |

There are 77,034 eligible rows across 182 usable signal dates. Ratios are approximately 50/10/10/30 by signal date, not random stock rows. The classification test uses 5,394 extreme-label rows, whereas inference retains all 8,985 eligible test rows.

### How parameters are selected

Each model has three predefined settings. Three expanding chronological folds within training minimize mean log loss; separate validation loss is a tie-break. Training labels whose realization extends to or beyond the next evaluation signal are purged. Winsorization, imputation and scaling are fitted only on the relevant training fold. Selected models refit on 23,014 purged train-plus-validation labels. Test and later backtest returns do not choose parameters. See `results/cv_search.csv`, `selected_parameters.json` and `splits.json`.

## 5 Main results and their meaning

The common portfolio period is **2020-10-20 to 2025-04-30**, after primary trading costs. The benchmark is the CSI 500 price index, not a dividend-reinvested total-return series.

| Model | Test accuracy | Test AUC | Total return | Annual return | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 52.02% | 0.532 | -1.50% | -0.35% | 0.110 | 37.24% |
| Random Forest | **54.56%** | **0.555** | 60.04% | 11.59% | 0.683 | **17.34%** |
| XGBoost | 54.23% | 0.553 | 58.47% | 11.33% | 0.674 | 19.20% |
| LightGBM | 54.00% | 0.550 | **80.21%** | **14.72%** | **0.816** | 19.80% |
| CSI 500 | Not applicable | Not applicable | -7.32% | -1.76% | 0.035 | 42.19% |

![Daily portfolio net values and aligned drawdowns](figures/daily_nav_drawdown.png)

Random Forest has the strongest reported held-out classification metrics and lower primary-portfolio drawdown; LightGBM has the highest realized primary-portfolio return in this later period. Those rankings are not contradictory: classification evaluates all labeled extremes, while a Top 10 portfolio depends on only the highest scores, return magnitudes, trading costs and risk. Differences are not established as statistically significant. Choosing a deployment winner after seeing these backtests would require another untouched future evaluation.

The old 665.32% XGBoost figure is not a course result. Corrected sample eligibility, information timing, model selection and the different evaluation window materially change the experiment. Neither the old numbers nor the paper's 153.64% return can be compared directly on an equal-period basis.

## 6 Environment and commands

Python **3.11** is the verified numerical environment. Run commands from this directory, not the repository root. For Windows PowerShell:

```powershell
cd course_project_mh6805
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install -r requirements-report.txt
python -m unittest discover -s tests -v
python tools/check_saved_results.py
```

On macOS/Linux use `python3.11 -m venv .venv` and `source .venv/bin/activate`; subsequent commands are the same. Unit tests use deterministic fixtures, not investable sample data. `check_saved_results.py` only checks the published aggregate snapshot.

After obtaining authorized raw inputs described in [DATA_ACCESS.md](docs/DATA_ACCESS.md):

```bash
python tools/check_inputs.py
python run_project.py --stage all
python verify_results.py
python build_report.py
```

Unique scientific entry point: **`run_project.py`**. Individual stages are `prepare`, `models`, `backtest`, `report`. `all` performs all four in order. `models` includes temporal selection and ablations. `verify_results.py` needs locally regenerated stock-level data. `build_report.py` only formats existing computed evidence into PDF and LaTeX; it is not a backtest and does not regenerate Word/PPTX. Rerunning stages overwrites the corresponding outputs, so use a separate working clone for experiments.

The optional PDF builder uses Windows Times New Roman if available and PDF Times elsewhere. It has no dependency on an author's home directory. Supplied PPTX and DOCX are editable without the private artifact-authoring toolchain. For Overleaf, upload the contents of `report_latex/`, select `main.tex` and compile; verify pagination after changes. The uploaded source compiled successfully to seven pages during preparation.

## 7 Directory and key outputs

```text
run_project.py                 scientific entry point
config/research.json           dates, costs, sampling and fixed seed
src/data.py                    universe, factor availability and targets
src/modeling.py                fold-fitted preprocessing and four estimators
src/backtest.py                daily ledgers and diagnostic hedge overlays
src/reporting.py               charts, evidence summaries and HTML
tests/                         deterministic unit tests
tools/                         input and saved-evidence checks
data/raw/                      private provider caches supplied locally
data/processed/                regenerated stock-level panels, ignored
results/                       aggregate published CSV/JSON evidence
figures/                       PNG and vector PDF charts
deliverables/                  Word, PDF, slides, notes and offline HTML
report_latex/                  editable report source and figures
docs/                          bilingual guides, detailed rules and caveats
```

`prediction_metrics.csv` contains classification/ranking metrics by model and split; `portfolio_metrics.csv` contains long-only and diagnostic-hedge results; `daily_nav.csv` contains daily aggregate curves; `decile_metrics.csv` and `decile_daily_nav.csv` contain score-group outcomes; `factor_ablations.csv` compares 40 fitted factor configurations; `permutation_importance.csv` records held-out loss sensitivity; `annual_performance.csv` and `cost_sensitivity.csv` describe regimes and costs. `verification.json` and `data_audit.json` describe the original local run and input hashes.

## 8 Limits and collaboration rules

Read [KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md). Historical revisions and constituent publication timing remain uncertain. Opening fills, price limits, lot sizes, market impact, dividend cash flows, borrowing/funding and intraday risk are incomplete. Short overlays are diagnostics, not executable stock-short/futures strategies. We report permutation importance, **not SHAP**. Factor ablations are exploratory held-out comparisons, not a new tuning opportunity.

For new experiments, create a branch, state the hypothesis first, preserve the original result snapshot and document changed settings. Do not use the current test/backtest to repeatedly optimize and still call it untouched. Never commit provider raw tables, `.env`, tokens, individual stock predictions or fitted pickle/joblib files. Access to a private cache is not permission to redistribute it.

## 9 This upload

The 2026-10-20 course packaging adds bilingual explanations, reading/data guides, public-only validation tools and the latest Word report. It does **not** retrain models or alter the course numerical results. The root internship archive remains unchanged except for navigation links.
