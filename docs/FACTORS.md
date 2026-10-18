# Factor dictionary

These 24 column names are read from the archived dataset metadata. They describe the implementation, not a claim of exact fidelity to the original paper.

| Family | Columns | Rationale |
|---|---|---|
| Valuation (4) | profit_to_market, book_to_market, sales_to_market, cashflow_to_market | Relative valuation and cash-flow proxy |
| Size (1) | log_mv | Size exposure |
| Momentum (2) | mom20, mom60 | Short/medium price trend |
| Market sensitivity (1) | beta | Rolling benchmark sensitivity |
| Liquidity (2) | turnover20, turnover60 | Trading intensity |
| Technical (2) | macd_close, mfi | Trend and money-flow indicators |
| Growth (3) | revenue_yoy, netprofit_yoy, ocf_yoy | Business/earnings/cash-flow growth |
| Quality (5) | roe, roa, grossprofit_margin, netprofit_margin, assets_turn | Profitability and asset efficiency |
| Leverage (2) | debt_to_assets, longdebt_to_assets | Liability exposure |
| Liquidity/solvency (2) | current_ratio, cash_ratio | Short-term balance-sheet resilience |

Exact formulas and fallback schema mappings: `src/utils/technical.py` and `src/utils/finance.py`. In particular, the cash-flow valuation proxy and liability denominator must be checked against source fields before comparing with textbook definitions. Accounting missingness, industry differences and correlated features are material caveats.
