# Remaining limitations

1. Announcement date controls do not certify historical statement vintages. Cached provider revisions may retain earlier announcement fields. The raw source lacks enough evidence to prove this fully absent.
2. Membership snapshot dates do not always identify the exact public announcement timestamp. A 62-day maximum staleness rule is explicit but not a complete index-event reconstruction.
3. Missing disclosures reduce early sample coverage. The filtering uses contemporaneous missingness, but selected firms may differ systematically from excluded firms.
4. OCF/price uses cumulative reported per-share cash flow, not a validated TTM yield. Noncurrent liabilities is broader than interest-bearing long-term debt. Momentum counts observed stock bars, which may span more exchange days during suspensions.
5. Prices use adjustment factors and fractional adjusted units, not an actual-share corporate-action/cash-dividend ledger. CSI 500 is a price index; its dividend treatment differs from a total-return benchmark.
6. Execution uses next-session observed opens with positive volume, without auction capacity, price limits, ST history, lot constraints or impact. A quoted opening price need not be fully executable.
7. Suspended assets carry last observed open for valuation and remain locked. This may hide interim economic losses. Delisted holdings use zero recovery, a conservative assumption rather than measured settlements.
8. The last holding period is truncated at the cached final open. No observations are fabricated beyond 2025-04-30.
9. The two 50/50 hedges are diagnostic daily return overlays, not full executable short-position ledgers. Funding, borrow restrictions, margin, index futures basis/roll, hedge drift and related rebalancing costs remain incomplete.
10. Models fit once before the test window, not on a walk-forward retraining schedule. Three candidates per model are finite search points, not global optima. The 18-date test period and numerous explanatory comparisons limit confidence.
11. Permutation importance is marginal predictive importance with correlated inputs, not causality. The project computes importance rather than SHAP; it does not claim SHAP results.
12. Raw observations are preserved; feature imputations are model transforms. Some unavailable-source values can only be represented as missing, not recovered by asserting a fresh download is complete.

The new results should be cited with these assumptions. Do not reuse the old 665.32% XGBoost number or label-filtered legacy plots in the course final report.
