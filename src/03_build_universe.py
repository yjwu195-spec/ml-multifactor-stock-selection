from __future__ import annotations

import pandas as pd

from utils.common import choose_existing_path, ensure_project_dirs, load_config, print_step, read_dataframe, resolve_path, save_dataframe


def main() -> None:
    config = load_config()
    ensure_project_dirs(config)
    raw_dir = resolve_path(config["paths"]["raw"])
    processed_dir = resolve_path(config["paths"]["processed"])

    rebalance = read_dataframe(processed_dir / "rebalance_calendar.csv", dtype={"rebalance_date": str})
    index_weight = read_dataframe(
        choose_existing_path(
            raw_dir / f"index_weight_{config['data']['benchmark_index'].replace('.', '_')}.parquet",
            raw_dir / f"index_weight_{config['data']['benchmark_index'].replace('.', '_')}.csv.gz",
        ),
        dtype={"trade_date": str, "con_code": str},
    )
    if index_weight.empty:
        raise ValueError("index_weight 数据为空，无法构建动态股票池。")

    index_weight = index_weight.rename(columns={"con_code": "ts_code"})
    index_weight = index_weight.sort_values(["trade_date", "ts_code"])

    results = []
    available_dates = sorted(index_weight["trade_date"].astype(str).unique().tolist())
    for rebalance_date in rebalance["rebalance_date"].astype(str):
        valid_dates = [date for date in available_dates if date <= rebalance_date]
        if not valid_dates:
            continue
        effective_date = valid_dates[-1]
        slice_df = index_weight[index_weight["trade_date"] == effective_date].copy()
        slice_df["rebalance_date"] = rebalance_date
        slice_df["effective_weight_date"] = effective_date
        results.append(slice_df[["rebalance_date", "effective_weight_date", "ts_code", "weight"]])

    universe = pd.concat(results, ignore_index=True)
    save_dataframe(universe, processed_dir / "dynamic_universe.parquet")

    min_count = universe.groupby("rebalance_date")["ts_code"].nunique().min()
    if min_count < config["universe"]["min_constituents_warning"]:
        print_step(f"警告：某些调仓日成分股数量较低，最小值为 {min_count}")
    print_step(f"动态股票池已生成，共 {universe['rebalance_date'].nunique()} 个调仓日。")


if __name__ == "__main__":
    main()
