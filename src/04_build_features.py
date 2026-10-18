from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.common import (
    choose_existing_path,
    concat_dataframes,
    concat_csv_files,
    ensure_project_dirs,
    load_config,
    print_step,
    read_dataframe,
    resolve_path,
    save_dataframe,
)
from utils.finance import align_financials_to_rebalance, prepare_balancesheet, prepare_fina_indicator
from utils.technical import compute_market_factors


def load_market_data(raw_dir: Path, benchmark_code: str):
    daily_parquet = sorted(raw_dir.glob("daily_*.parquet"))
    daily_basic_parquet = sorted(raw_dir.glob("daily_basic_*.parquet"))
    adj_parquet = sorted(raw_dir.glob("adj_factor_*.parquet"))
    daily = (
        concat_dataframes(daily_parquet, dtype={"ts_code": str, "trade_date": str})
        if daily_parquet
        else concat_csv_files(raw_dir.glob("daily_*.csv.gz"), dtype={"ts_code": str, "trade_date": str})
    )
    daily_basic = (
        concat_dataframes(daily_basic_parquet, dtype={"ts_code": str, "trade_date": str})
        if daily_basic_parquet
        else concat_csv_files(raw_dir.glob("daily_basic_*.csv.gz"), dtype={"ts_code": str, "trade_date": str})
    )
    adj_factor = (
        concat_dataframes(adj_parquet, dtype={"ts_code": str, "trade_date": str})
        if adj_parquet
        else concat_csv_files(raw_dir.glob("adj_factor_*.csv.gz"), dtype={"ts_code": str, "trade_date": str})
    )
    benchmark = read_dataframe(
        choose_existing_path(
            raw_dir / f"index_daily_{benchmark_code.replace('.', '_')}.parquet",
            raw_dir / f"index_daily_{benchmark_code.replace('.', '_')}.csv.gz",
        ),
        dtype={"trade_date": str},
    )
    return daily, daily_basic, adj_factor, benchmark


def merge_market_panel(daily: pd.DataFrame, daily_basic: pd.DataFrame, adj_factor: pd.DataFrame) -> pd.DataFrame:
    cols_needed = ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"]
    daily = daily[cols_needed].copy()
    basic_keep = [col for col in ["ts_code", "trade_date", "turnover_rate", "total_mv", "pe_ttm", "pb", "ps_ttm", "pcf_ocf_ttm"] if col in daily_basic.columns]
    panel = daily.merge(daily_basic[basic_keep], on=["ts_code", "trade_date"], how="left")
    panel = panel.merge(adj_factor[["ts_code", "trade_date", "adj_factor"]], on=["ts_code", "trade_date"], how="left")
    panel["adj_factor"] = panel["adj_factor"].fillna(1.0)
    return panel


def build_valuation_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_mv"] = np.log(out["total_mv"].replace({0: np.nan}))
    out["profit_to_market"] = 1 / out["pe_ttm"].replace({0: np.nan})
    out["book_to_market"] = 1 / out["pb"].replace({0: np.nan})
    out["sales_to_market"] = 1 / out["ps_ttm"].replace({0: np.nan})
    if "pcf_ocf_ttm" in out.columns:
        out["cashflow_to_market"] = 1 / out["pcf_ocf_ttm"].replace({0: np.nan})
    else:
        out["cashflow_to_market"] = np.nan
    return out


def main() -> None:
    config = load_config()
    ensure_project_dirs(config)
    raw_dir = resolve_path(config["paths"]["raw"])
    processed_dir = resolve_path(config["paths"]["processed"])
    feature_dir = resolve_path(config["paths"]["features"])

    universe = read_dataframe(processed_dir / "dynamic_universe.parquet", dtype={"rebalance_date": str, "ts_code": str})
    daily, daily_basic, adj_factor, benchmark = load_market_data(raw_dir, config["data"]["benchmark_index"])
    print_step("合并市场面板数据")
    market_panel = merge_market_panel(daily, daily_basic, adj_factor)
    market_panel = compute_market_factors(
        market_panel,
        benchmark,
        beta_window=config["data"]["lookback_beta"],
        mfi_window=config["data"]["lookback_mfi"],
    )
    market_panel = build_valuation_features(market_panel)
    save_dataframe(market_panel, processed_dir / "market_panel_daily.parquet")

    rebalance_market = market_panel.merge(
        universe[["rebalance_date", "ts_code"]],
        left_on=["trade_date", "ts_code"],
        right_on=["rebalance_date", "ts_code"],
        how="inner",
    )
    rebalance_market = rebalance_market.drop(columns=["rebalance_date"]).rename(columns={"trade_date": "rebalance_date"})

    fina_parquet = sorted(raw_dir.glob("fina_indicator_part_*.parquet"))
    bs_parquet = sorted(raw_dir.glob("balancesheet_part_*.parquet"))
    fina_df = prepare_fina_indicator(
        concat_dataframes(fina_parquet, dtype={"ts_code": str})
        if fina_parquet
        else concat_csv_files(raw_dir.glob("fina_indicator_part_*.csv.gz"), dtype={"ts_code": str})
    )
    bs_df = prepare_balancesheet(
        concat_dataframes(bs_parquet, dtype={"ts_code": str})
        if bs_parquet
        else concat_csv_files(raw_dir.glob("balancesheet_part_*.csv.gz"), dtype={"ts_code": str})
    )
    financial_panel = align_financials_to_rebalance(universe, fina_df, bs_df)

    features = universe.merge(
        rebalance_market[
            [
                "rebalance_date",
                "ts_code",
                "profit_to_market",
                "book_to_market",
                "sales_to_market",
                "cashflow_to_market",
                "log_mv",
                "mom20",
                "mom60",
                "beta",
                "turnover20",
                "turnover60",
                "macd_close",
                "mfi",
            ]
        ],
        on=["rebalance_date", "ts_code"],
        how="left",
    )
    features = features.merge(
        financial_panel[
            [
                "rebalance_date",
                "ts_code",
                "revenue_yoy",
                "netprofit_yoy",
                "ocf_yoy",
                "roe",
                "roa",
                "grossprofit_margin",
                "netprofit_margin",
                "assets_turn",
                "debt_to_assets",
                "longdebt_to_assets",
                "current_ratio",
                "cash_ratio",
            ]
        ],
        on=["rebalance_date", "ts_code"],
        how="left",
    )
    save_dataframe(features, feature_dir / "feature_panel.parquet")
    print_step(f"因子面板已生成，共 {len(features)} 行。")


if __name__ == "__main__":
    main()
