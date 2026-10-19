# Detailed methods and data rules

## Information clock

At a signal close t, select the latest dated CSI 500 membership snapshot no later than t and at most 62 calendar days old. Generate market features only from bars up to t. Fundamentals require available_date <= t; available_date is announcement date plus one day for indicators, and max(ann_date, f_ann_date) plus one day for balance sheets. The first trading session after t is the execution date. Next-period labels run from that execution open to the following execution open.

Statement events are sorted by availability and report end. Ignore late filings for an older report period once a newer period is available. Keep the first deterministic vendor record for identical code/announcement/report keys and report their duplicate count; this rule is not proof of original-vintage correctness. A financial record becomes unavailable after 550 days since release. Missing values remain missing until the fold-fitted feature imputer, with no filling of raw price/statement observations.

## Eligibility and missing observations

Require an actual current quote with positive volume and close, at least 60 observed historical records, a computable 60-record momentum return, and at most 40% missing model features. This rule permits an unbalanced panel; do not backfill nonlisted history or require survival. All models use this same eligibility panel. Training classification uses only labeled extremes; inference uses every eligible row regardless of future label/return availability.

Missing entry/exit quotes make a classification/ranking target unavailable. They do not remove that stock from scoring or a contemporaneous investment decision. In the trading ledger unavailable opening entries leave cash instead of replacing them by a lower-ranked security using future information. Holdings that cannot be traded remain locked; the last genuine observed open marks their value. Raw missing quotes are unchanged. Known delisted holdings get a conservative zero recovery at their actual delisting date. This settlement assumption is not a measured realized recovery.

## Factor definitions

| Family | Factors and definitions |
|---|---|
| Valuation | earnings_price = 1/pe_ttm; book_price = 1/pb; sales_price = 1/ps_ttm; ocf_price = latest available reported ocfps / current unadjusted close |
| Growth | revenue_yoy = or_yoy; netprofit_yoy; ocf_yoy, all from latest eligible financial report |
| Quality | roe; roa; grossprofit_margin; netprofit_margin; assets_turn |
| Leverage/liquidity | debt_to_assets; noncurrent_liabilities_assets = total_ncl / total_assets; current_ratio; cash_ratio |
| Momentum | adjusted close returns over 20 and 60 observed stock bars |
| Trading/risk | 20/60-bar mean turnover; 60-bar rolling beta against matched-day CSI 500 close return; normalized MACD histogram; 14-bar MFI; annualized 20-bar close-return volatility |
| Size | log(total_mv) |

No missing adjustment factor is silently replaced by 1.0. Adjusted price = observed unadjusted price times observed adjustment factor. Factor cash-flow-to-price is a latest-reported cumulative cash-flow proxy, not certified TTM cash-flow yield; different report lengths and share changes are limitations. Noncurrent liabilities is not identical to interest-bearing long-term debt. Ratios inherit provider units, with zero denominators treated as missing. Values in percent are not mixed into return arithmetic as fractions.

## Labels and transforms

Forward excess return = stock adjusted-open return minus CSI 500 unadjusted price-index open return. At each date, lower 30% is class 0, upper 30% class 1. Ties use inclusive quantiles; the recorded sample describes resulting counts. Middle samples remain eligible inference rows. A model score is conditional extreme-group discrimination, not a calibrated unconditional benchmark-outperformance probability.

Fold-fitted transforms: clip each feature to training 1%/99% quantiles; train median imputation (empty training columns transform to constant zero); StandardScaler. The same transform structure serves all models. Unavailable raw observations are not overwritten by imputed numbers. All fitted estimators, transforms and selected settings are saved.

## Temporal selection

Split the usable rebalance dates 50/10/10/remaining 30% chronologically. Three expanding TimeSeriesSplit folds inside training produce out-of-fold log losses. Purge training examples with label exit >= validation's first signal. Choose among three settings/model by mean time-CV loss; independent validation loss breaks ties. Refit selected estimators on purged train+validation labels before testing. No test/backtest metric chooses a parameter, model winner for future deployment, or factor configuration. Test ablations are descriptive and their strong variants must not be promoted without a new untouched evaluation sample.

## Portfolio accounting

The primary book starts with equity 1.0, holds fractional adjusted-price units, receives zero interest on cash and targets equal Top 10 values every 20 exchange sessions. Frozen holdings reduce freely allocatable equity. Execute sells before buys, debit one-way cost on each, and scale buys to avoid negative cash. Mark at each observed exchange open, not at a signal close. The last available open is the final valuation/liquidation point, even if it truncates the last 20-session interval.

Turnover = (absolute buys + absolute sells)/pre-trade equity; this includes both sides of replacement and respects drift. Initial entry and executable final liquidation pay costs. Primary cost =10 bps; sensitivity 0/5/10/20 bps. Ten deciles use the same trading ledger; score ties break deterministically by ts_code. Decile 10 is the highest score, not an inverse label group.

Diagnostic hedges allocate 50% of daily return to the Top-10 sleeve and subtract 50% of index/bottom-sleeve return. The index overlay pays 0.5 x1 bp on rebalance events; bottom-stock costs are charged to its half-sleeve. This is a stylized constant-notional return overlay, not a complete self-financing short-share ledger. Margin/borrow, hedge drift/rebalancing costs, index derivative basis and funding are incomplete. Do not describe these overlays as implemented short selling.

## Evaluation

Classification: accuracy, balanced accuracy, F1, AUC, log loss and confusion matrices on labeled extremes. Ranking: same-date Spearman score/excess-return correlation on observable outcomes, date-level mean/SD/IR, six-period circular block-bootstrap mean intervals with 2,000 replicates. Portfolio: net daily compounding, (ending NAV)^(252/days)-1, daily volatility x sqrt(252), zero-risk-free arithmetic daily Sharpe, and peak-to-trough daily-open drawdown including initial NAV 1.0. These are trading-session annualizations, not calendar-day CAGR. Annual slices are total within-slice returns; partial 2020/2025 are not full-year claims.

Interpretation: full-model permutation importance uses 1,500 fixed-seed real labeled test observations, neg_log_loss scoring, three repeats, and reports mean/SD. Correlated features can substitute and importance is not causal. Factor ablations use the full-model hyperparameters without retuning on test. Calendar-year predictive and portfolio tables document temporal instability. Multiple testing and regime selection remain limitations, not formal significance claims.
