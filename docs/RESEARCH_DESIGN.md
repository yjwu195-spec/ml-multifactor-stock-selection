# Research design and development history

## Research question

Can nonlinear tree ensembles combine 24 cross-sectional fundamental and market factors to predict the direction of subsequent CSI-500-relative returns, and how do model differences affect ranking and portfolio risk?

## Implemented sequence

1. Download real Tushare observations with retries, year batches and 80-stock financial chunks.
2. Build a trading calendar and one rebalance date every 20 open days.
3. At each rebalance date use the most recent `index_weight` snapshot with snapshot date <= decision date.
4. Derive adjusted-price, valuation, momentum, liquidity and financial factors; align statements backward by announcement date.
5. Compute stock next-period returns and subtract the CSI 500 return. Top 30% excess returns receive class 1, bottom 30% class 0; the middle is unlabeled.
6. Existing v2 code filters labeled rows and applies median imputation, MAD winsorization and cross-sectional standardization. This is also applied to the backtest sample and is a known critical issue, described in KNOWN_LIMITATIONS.md.
7. Divide rebalance dates chronologically 50% train / 10% validation / 10% test / remaining 30% backtest. Actual dates/row counts are in `results/reference/dataset_metadata_improved.json`.
8. Fit RF baseline and six configurations each for XGBoost/LightGBM on train. Choose boosting parameters by validation accuracy, lower log loss, then smaller train-validation accuracy gap. Refit chosen model on train+validation; held-out metrics are descriptive only.
9. Rank probabilities; evaluate ten groups and equal-weight Top 10 long-only, Top 10 minus Bottom 10, and Top 10 minus CSI 500.
10. Deduct 10 bps stock turnover costs, plus a 1 bp per-period index hedge cost when applicable. Calculate period NAV, annualized returns, volatility, zero-risk-free Sharpe and drawdown.

## Model comparison

Random Forest averages bootstrapped trees to reduce variance. XGBoost adds trees sequentially to optimize a regularized loss; shallow trees, low learning rates, subsampling and penalties constrain complexity. LightGBM uses efficient histogram-based boosting and leaf-wise growth; depth/leaves and minimum leaf sample counts constrain overfitting. A stronger classifier does not necessarily produce better Top-N portfolios because ranking errors, extreme outcomes and turnover matter.

## Development stages

| Stage | Change | Interpretation |
|---|---|---|
| Original replication | RF, factor and forward-label pipeline | Practical adaptation of broker research |
| Data engineering | Resumable downloads, large panels in Parquet | Reduces I/O cost; does not guarantee data completeness |
| Model extension | Added XGBoost/LightGBM | Tests nonlinear ensemble differences |
| Exploratory search | Early search selected by backtest Sharpe | Selection bias; old 665.32% figure excluded from main snapshot |
| Improved v2 | Separate validation set, six conservative candidates | Better parameter selection separation; other leakage remains |
| Portfolio analysis | Trading costs, deciles, long-short and index hedge | Separates ranking and portfolio behavior |
| Drawdown interpretation | Distinguished deciles from bottom-stock shorts | Very large losses belong to long-short variant, not automatically all deciles |
| GitHub packaging | Relative paths, safe credentials, docs/tests and archived outputs | Code portability and disclosure, not a new research run |

## Planned rigorous continuation

Correct the label-independent inference universe and preprocessing, purge label horizons, validate point-in-time fundamentals, add daily NAV and tradability rules, then perform a fresh walk-forward study with a frozen untouched test interval. New corrected results must be stored separately from the archived snapshot.
