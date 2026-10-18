from __future__ import annotations

import pandas as pd

from utils.common import choose_existing_path, ensure_project_dirs, load_config, print_step, read_dataframe, resolve_path, save_dataframe


def main() -> None:
    config = load_config()
    ensure_project_dirs(config)
    processed_dir = resolve_path(config["paths"]["processed"])
    label_dir = resolve_path(config["paths"]["labels"])
    raw_dir = resolve_path(config["paths"]["raw"])

    universe = read_dataframe(processed_dir / "dynamic_universe.parquet", dtype={"rebalance_date": str, "ts_code": str})
    rebalance = read_dataframe(processed_dir / "rebalance_calendar.csv", dtype={"rebalance_date": str, "next_rebalance_date": str})
    market_panel = read_dataframe(processed_dir / "market_panel_daily.parquet", dtype={"trade_date": str, "ts_code": str})
    benchmark = read_dataframe(
        choose_existing_path(
            raw_dir / f"index_daily_{config['data']['benchmark_index'].replace('.', '_')}.parquet",
            raw_dir / f"index_daily_{config['data']['benchmark_index'].replace('.', '_')}.csv.gz",
        ),
        dtype={"trade_date": str},
    )

    benchmark = benchmark.sort_values("trade_date")
    benchmark_prices = benchmark[["trade_date", "close"]].rename(columns={"trade_date": "rebalance_date", "close": "bench_close"})
    benchmark_next = benchmark_prices.rename(columns={"rebalance_date": "next_rebalance_date", "bench_close": "bench_close_next"})

    stock_prices = market_panel[["trade_date", "ts_code", "adj_close"]].rename(columns={"trade_date": "rebalance_date", "adj_close": "stock_close"})
    stock_next = stock_prices.rename(columns={"rebalance_date": "next_rebalance_date", "stock_close": "stock_close_next"})

    labels = universe.merge(rebalance[["rebalance_date", "next_rebalance_date"]], on="rebalance_date", how="left")
    labels = labels.merge(stock_prices, on=["rebalance_date", "ts_code"], how="left")
    labels = labels.merge(stock_next, on=["next_rebalance_date", "ts_code"], how="left")
    labels = labels.merge(benchmark_prices, on="rebalance_date", how="left")
    labels = labels.merge(benchmark_next, on="next_rebalance_date", how="left")

    labels["stock_ret_fwd"] = labels["stock_close_next"] / labels["stock_close"] - 1
    labels["bench_ret_fwd"] = labels["bench_close_next"] / labels["bench_close"] - 1
    labels["excess_ret"] = labels["stock_ret_fwd"] - labels["bench_ret_fwd"]

    top_pct = config["label"]["top_pct"]
    bottom_pct = config["label"]["bottom_pct"]

    stats = (
        labels.groupby("rebalance_date")["excess_ret"]
        .agg(
            valid_count="count",
            upper=lambda s: s.quantile(1 - top_pct),
            lower=lambda s: s.quantile(bottom_pct),
        )
        .reset_index()
    )
    labels = labels.merge(stats, on="rebalance_date", how="left")
    labels["label"] = pd.NA
    enough_sample = labels["valid_count"] >= 10
    labels.loc[enough_sample & (labels["excess_ret"] >= labels["upper"]), "label"] = 1
    labels.loc[enough_sample & (labels["excess_ret"] <= labels["lower"]), "label"] = 0
    labels = labels.drop(columns=["valid_count", "upper", "lower"])
    save_dataframe(labels, label_dir / "label_panel.parquet")
    print_step("标签面板已生成。")


if __name__ == "__main__":
    main()
