# Known limitations and interpretation rules

Recorded during packaging on 2026-10-19. Source behavior is preserved for historical traceability; these issues were not repaired by silently altering the reported results.

## Critical: future-label-dependent backtest universe

`src/improved_v2/01_make_dataset_with_validation.py` filters the merged panel with `dataset["label"].isin([0, 1])` before assigning all four splits. The label is defined by forward excess returns. The backtest consequently selects stocks from an ex-post subset containing only the best/worst 30%, excluding the middle 40% using information unavailable on the decision date. Model probabilities do not remove this bias. Ten-decile membership is also affected.

Required future correction: preserve the full date-specific investable feature panel for inference; only training/validation classification samples may use extreme-return labels. Evaluate portfolio returns on the complete inference universe. Rebuild preprocessing and all results afterward. The direction/magnitude of the corrected performance cannot be inferred from the archived numbers.

## Preprocessing leakage and missing data

`cross_section_fillna` uses a per-date median, then a median over the entire supplied panel as a fallback. The dataset code preprocesses before chronological splitting. Missing values for an all-missing date can therefore depend on future dates. Per-date statistics are also calculated after ex-post label filtering. Raw missing observations are not preserved in the modeled panel. A future repair should use a documented train-fitted imputation policy and retain raw missingness/provenance; never manufacture market observations.

## Label horizon and chronology

Date-order splits do not explicitly purge examples whose next rebalance date crosses a split boundary. A future implementation must separate label horizons and clarify when the first test/backtest trade is allowed after the final model fit. All three models have fixed fits rather than a walk-forward retraining schedule.

## Execution and drawdown

- Rebalance-date closing features and closing-price returns imply idealized execution timing; no order-level fill model is provided.
- Drawdown uses rebalance-frequency NAV, not daily mark-to-market. It may miss intraperiod losses.
- The existing `nav / nav.cummax() - 1` formula omits initial capital 1.0 from the running maximum. An initial loss can be understated; fix and rerun before relying on risk metrics.
- Long-short return spreads imply approximately 100% long plus 100% short exposure, not a 50/50 fully funded allocation. Index hedging is theoretical, not an implemented futures execution strategy.
- Stock costs use absolute equal-weight target turnover. Weight drift, financing, margin, borrow fees/availability, stamp-duty changes, futures basis/roll, suspensions, price limits and capacity are not fully modeled.
- Decile curves are simple gross mean returns; stock transaction costs apply to the three portfolio variants, not automatically to all decile curves.

## Data and model caveats

- Historical membership snapshots are as-of dated; publication timing and gaps still need auditing.
- Financial observations are aligned backward by announcement date, but restatement vintage and after-close announcements are not fully resolved.
- Endpoint truncation, historical delisted-stock coverage and empty download responses require completeness audits; a saved file does not prove completeness.
- LightGBM `subsample` is set without enabling a positive `subsample_freq`; its row-bagging effect should not be assumed.
- Parameter search contains six candidates per boosting model, not an exhaustive/global optimum. RF is a configured baseline, not a similarly sized tuning search.
- Classification accuracy on extreme-label samples is not directly comparable to investing performance across all constituents.
- Dependency version ranges do not recreate the exact original training environment.

## Archived deliverables

Word, slides and PNGs were copied as historical artifacts and were not reauthored in this release. Their recommendations and headline performance predate these disclosures. `results/reference` and the HTML overview record historical data, not new verification of investment performance. None of the packaging checks substitutes for a fresh isolated-data full backtest.
