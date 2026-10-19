# Response to proposal feedback

Group 12: Wu Yangjie, Zhao Jingjie, Wang Zhaobo. All contributed equally.

| Feedback | Implemented response | Evidence |
|---|---|---|
| Direct Tushare/API links missing | Added direct endpoint links to report references, README and dataset notes | Final report p.7; README |
| Actual fundamental release dates | Announcement-day plus one calendar-day availability; later ann/f_ann date for balance sheets; report end is never availability | data.py financial_alignment and prepare; unit tests |
| Unbalanced panel, new/suspended stocks | Current-date eligibility, sufficient observed history, financial staleness and missingness cutoff; unavailable entries stay in cash, untradeable exits stay locked | data_audit.json, universe_coverage.csv, daily_nav.csv |
| Identical rebalance dates | All four models score exactly the same full eligible stock-date keys and use a common daily-open portfolio window | verify_results.py; verification.json |
| Factor-family ablations | All, fundamentals only, market only, and seven leave-one-family-out variants for each model; frozen full-model parameters | factor_ablations.csv; ablation_portfolios.csv; 40 rows each |
| SHAP/importance | Held-out permutation log-loss importance with fixed actual test subset, 3 repeats, mean/std; no SHAP claim | permutation_importance.csv; figure |
| Subperiod analysis | Calendar-year portfolio outcomes and predictive/Rank-IC metrics by split/year; partial boundary years identified | annual_performance.csv; prediction_by_year.csv |
| Prediction/ranking plus portfolios | Accuracy, balanced accuracy, F1, AUC, log loss, confusion matrices, Rank IC, block bootstrap and ten score groups | prediction_metrics.csv, rank_ic.csv, rank_ic_bootstrap.csv, decile outputs |
| Turnover and transaction costs | Actual ledger turnover with drift, 0/5/10/20-bps sensitivity, main 10-bps cost; normalized hedged diagnostics separately identified | cost_sensitivity.csv; daily_nav.csv; portfolio_metrics.csv |

## Additional corrections from the legacy implementation

- Future-return labels cannot filter the inference or investment universe.
- Preprocessing statistics are fitted only on training data, inside each CV fold.
- Labels crossing into a later evaluation signal are purged from training.
- Open-after-signal execution avoids using a closing signal to buy at that same close.
- Daily-open drawdown includes starting capital 1.0.
- An actual reported OCF-per-share/price proxy replaces the unavailable legacy pcf field; its cumulative accounting period definition is disclosed.
- Realized daily portfolios replace old archived performance statistics; no old result is silently presented as a new run.

## Interpretation boundary

The announcement-lag rule cannot establish that cached statement values have certified historical revision vintages. This is unresolved data provenance, disclosed in the report, rather than an asserted complete elimination of all future-information risk. The project is research under explicit assumptions, not a validated trading product.
