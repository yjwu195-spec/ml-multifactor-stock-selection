# Group 12 presentation notes

## Slide 1 Machine Learning Based Multifactor Stock Selection

Introduce the project as a test of nonlinear ranking rather than a promise of investment profit. Wu Yangjie, Zhao Jingjie and Wang Zhaobo contributed equally. Data source: Tushare Pro, https://tushare.pro/document/2.

## Slide 2 Research questions

Explain that we compare linear versus nonlinear prediction, complementary information families, and economic performance after costs. Methods references: Breiman 2001; Chen and Guestrin 2016 https://arxiv.org/abs/1603.02754; Ke et al. 2017 https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.

## Slide 3 Data coverage and information availability

Raw sample: 2010-01-04 to 2025-04-30. Sources: daily https://tushare.pro/document/2?doc_id=27; daily_basic doc_id=32; fina_indicator doc_id=79; balancesheet doc_id=78; trade_cal doc_id=26; index_weight doc_id=96. The actual announcement date, not period end, governs availability. Financial vintage limitations remain.

## Slide 4 Chronological validation and leakage controls

Training dates 2010-05-05 through 2017-09-26, validation 2017-10-31 through 2019-03-25, test 2019-04-23 through 2020-09-11 and later backtest 2020-10-19 through 2025-04-01 signal dates. Execution dates are one exchange session later. Purge training labels that are not known before the next evaluation signal. All preprocessing is fitted within each training fold.

## Slide 5 Model specification and complexity

Three candidate settings per model and three chronological CV folds; minimum mean log loss selects parameters with independent validation loss as tie-break. Saved selected_parameters.json gives exact settings. We activate LightGBM subsample_freq=1. A finite search is not a global optimum.

## Slide 6 Independent test prediction

Metrics apply to top/bottom 30% return labels, not all stocks. Test Rank IC uses the full eligible cross-section with observable forward returns. Classification scores are not calibrated unconditional probabilities of beating CSI 500.

## Slide 7 Ranking across ten portfolios

Group 1 contains the lowest scores, group 10 the highest. Every model uses the same stock-date universe. All decile ledgers pay 10 bps one-way turnover costs. Smooth ranking, not perfect monotonicity, is the relevant diagnostic. Source: results/decile_metrics.csv.

## Slide 8 Daily NAV and aligned drawdown

Primary long-only portfolios trade the Top 10 at the next session open. Daily open marks include transaction costs and actual weight drift. For readability this slide plots monthly sampled points from the daily curve; all daily observations are supplied in CSV and in the report figure. The benchmark uses the same dates. Suspension valuation uses the last observed opening quote. Source: results/daily_nav.csv.

## Slide 9 Portfolio outcomes after costs

The common portfolio window runs from 2020-10-20 to 2025-04-30 using daily opening marks. Main portfolio: equal-value Top 10, 10 bps per absolute traded notional, no substitute purchase when an opening order is unavailable. Sharpe uses zero risk-free rate. These results are newly computed, not the earlier biased archived statistics.

## Slide 10 Sensitivity to transaction costs

Four cost levels were fixed before evaluating portfolios: zero, five, ten and twenty basis points per one-way traded notional. The curves are descriptive robustness checks, not parameter selection criteria. Source: results/cost_sensitivity.csv.

## Slide 11 Factor family ablations

Each model is refit with the selected full-factor hyperparameters, using identical training samples and test dates. The test set is used for interpretation, not choosing a replacement factor set. Full outputs include seven leave-one-family-out runs plus the two information-family-only comparisons. Source: results/factor_ablations.csv.

## Slide 12 Held out feature importance in Random Forest

Permutation importance is the increase in test log loss after shuffling a feature, using 1,500 fixed-seed test observations and three repeats. Negative or near-zero values indicate weak marginal evidence; correlated substitutes can mask each other. These are predictive associations, not causal effects. SHAP was not computed because explicit importance satisfies the requested explanation alternative. Full four-model values: results/permutation_importance.csv.

## Slide 13 Stability across calendar periods

The first and last calendar slices are partial years. These subperiods are fixed calendar boundaries, not dates selected to maximize returns. Different market regimes and concentrated portfolio tails explain why classification and investment rankings can disagree. Source: results/annual_performance.csv.

## Slide 14 Conclusions and remaining limitations

Summarize modest nonlinear prediction gains, different model rankings for portfolios, and meaningful turnover/time sensitivity. Remaining limits: financial vintage revisions, index publication timing, auction execution, market impact, lot sizes, price limits and shorting feasibility. Hedged results are normalized diagnostic overlays, not executable futures trades. Wu Yangjie, Zhao Jingjie and Wang Zhaobo contributed equally. Do not present historical performance as a future guarantee.
