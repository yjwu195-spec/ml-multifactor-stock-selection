# Data contract and acquisition

No licensed raw market or financial tables are shipped. A reader must obtain their own Tushare permissions and download the configured history. Local download data is ignored by Git. Failure to obtain access is not replaced by synthetic data.

| Endpoint | Primary fields used | Purpose |
|---|---|---|
| trade_cal | exchange, cal_date, is_open, pretrade_date | Trading/rebalance calendar |
| stock_basic | ts_code, name, list_date, delist_date, industry | Stock directory and download batches |
| daily | ts_code, trade_date, open, high, low, close, pre_close, vol, amount | Prices and technical factors |
| daily_basic | ts_code, trade_date, pe/pe_ttm, pb, ps/ps_ttm, total_mv, turnover_rate | Valuation, size, liquidity |
| adj_factor | ts_code, trade_date, adj_factor | Adjusted-price derivation |
| index_daily | ts_code, trade_date, close | CSI 500 benchmark returns |
| index_weight | index_code, con_code, trade_date, weight | Historical membership snapshots |
| fina_indicator | ts_code, ann_date, end_date, growth/profitability/solvency fields | Financial factors |
| balancesheet | ts_code, ann_date, end_date, total_ncl, total_assets | Long-term liability ratio |

The code requests real endpoint observations. Provider units and schema must be checked against the Tushare documentation before economic interpretation. Some factor names are proxies rather than an exact original-paper definition (see factor dictionary).

## Files

- Raw foundation: `trade_cal.csv.gz`, `stock_basic.csv.gz`, `index_daily_000905_SH.csv.gz`, `index_weight_000905_SH.csv.gz`.
- Raw market: `daily_<year>.csv.gz`, `daily_basic_<year>.csv.gz`, `adj_factor_<year>.csv.gz`.
- Raw financials: `fina_indicator_part_<NNN>.csv.gz`, `balancesheet_part_<NNN>.csv.gz`.
- `src/00_convert_to_parquet.py` creates Parquet versions of large tables without removing originals. Foundation gzip CSV files are still read by the calendar/downloader.
- Processed: `rebalance_calendar.csv`, `dynamic_universe.parquet`, `market_panel_daily.parquet` and `*_dataset_improved.parquet`.
- Features: `feature_panel.parquet`, unique intended key `(rebalance_date, ts_code)`.
- Labels: `label_panel.parquet`, includes `next_rebalance_date`, forward stock/benchmark/excess returns and label.

Financial schema alternatives are resolved in `src/utils/finance.py`; market definitions are in `src/utils/technical.py`. Never rename files ad hoc without updating readers.

## Completeness checklist for a new run

Validate duplicates and date coverage, count constituents by rebalance date, investigate stale/missing membership snapshots, check all expected years and financial chunks, inspect missing factors and verify no API pagination/truncation. Confirm labels use actual next-period prices and account for delisted stocks. These are required research audits, not assertions that the archived data already passed them.

## Publication boundary

Only aggregate research outputs are included in `results/reference`. Do not commit `.env`, raw/processed panels, full scored stock panels or pretrained models by force. The original broker PDF is not redistributed. Before publishing archived internship documents, confirm that you have permission to release their contents. No exact historical data snapshot is distributed, so regeneration may differ due to provider revisions.
