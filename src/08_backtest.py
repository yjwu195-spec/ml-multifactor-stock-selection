from __future__ import annotations

from joblib import load
import pandas as pd

from utils.common import ensure_project_dirs, load_config, print_step, read_dataframe, read_json, resolve_path, save_dataframe


def get_model_path(output_dir):
    model_path = output_dir / "model.joblib"
    if model_path.exists():
        return model_path
    legacy_path = output_dir / "random_forest_model.joblib"
    if legacy_path.exists():
        return legacy_path
    raise FileNotFoundError("未找到模型文件，请先运行 07_train_model.py。")


def calculate_turnover_costs(holdings, cost_rate: float) -> pd.DataFrame:
    rows = []
    previous_weights: dict[str, float] = {}

    for rebalance_date, group in holdings.groupby("rebalance_date", sort=True):
        target_weight = 1.0 / len(group)
        current_weights = dict(zip(group["ts_code"], [target_weight] * len(group)))
        all_codes = set(previous_weights) | set(current_weights)
        turnover = sum(abs(current_weights.get(code, 0.0) - previous_weights.get(code, 0.0)) for code in all_codes)
        rows.append(
            {
                "rebalance_date": rebalance_date,
                "turnover": turnover,
                "transaction_cost": turnover * cost_rate,
            }
        )
        previous_weights = current_weights

    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    ensure_project_dirs(config)
    processed_dir = resolve_path(config["paths"]["processed"])
    output_dir = resolve_path(config["paths"]["outputs"])

    backtest_df = read_dataframe(processed_dir / "backtest_dataset.parquet", dtype={"rebalance_date": str, "ts_code": str})
    metadata = read_json(output_dir / "dataset_metadata.json")
    feature_cols = metadata["feature_columns"]
    model = load(get_model_path(output_dir))

    scored = backtest_df.copy()
    scored["score"] = model.predict_proba(scored[feature_cols])[:, 1]
    scored = scored.sort_values(["rebalance_date", "score"], ascending=[True, False])

    top_n = config["model"]["top_n"]
    holdings = scored.groupby("rebalance_date", group_keys=False).head(top_n).copy()
    holdings["target_weight"] = holdings.groupby("rebalance_date")["ts_code"].transform(lambda x: 1.0 / len(x))
    cost_rate = config["backtest"]["transaction_cost_bps"] / 10000.0
    turnover_costs = calculate_turnover_costs(holdings, cost_rate)

    period_returns = (
        holdings.groupby("rebalance_date")
        .agg(
            period_stock_ret=("stock_ret_fwd", "mean"),
            period_bench_ret=("bench_ret_fwd", "mean"),
            holdings_count=("ts_code", "count"),
            next_rebalance_date=("next_rebalance_date", "first"),
        )
        .reset_index()
    )
    period_returns = period_returns.merge(turnover_costs, on="rebalance_date", how="left")
    period_returns["turnover"] = period_returns["turnover"].fillna(0.0)
    period_returns["transaction_cost"] = period_returns["transaction_cost"].fillna(0.0)
    period_returns["period_stock_ret_net"] = period_returns["period_stock_ret"] - period_returns["transaction_cost"]
    period_returns["period_excess_ret_net"] = period_returns["period_stock_ret_net"] - period_returns["period_bench_ret"]

    nav = period_returns[["rebalance_date", "next_rebalance_date"]].copy()
    nav["strategy_nav"] = (1 + period_returns["period_stock_ret_net"]).cumprod() * config["backtest"]["initial_nav"]
    nav["benchmark_nav"] = (1 + period_returns["period_bench_ret"]).cumprod() * config["backtest"]["initial_nav"]
    nav["excess_nav"] = nav["strategy_nav"] / nav["benchmark_nav"]

    save_dataframe(scored, output_dir / "backtest_scored_panel.parquet")
    save_dataframe(holdings, output_dir / "backtest_holdings.parquet")
    save_dataframe(period_returns, output_dir / "backtest_period_returns.csv")
    save_dataframe(nav, output_dir / "backtest_nav.csv")
    print_step("回测结果已生成。")


if __name__ == "__main__":
    main()
