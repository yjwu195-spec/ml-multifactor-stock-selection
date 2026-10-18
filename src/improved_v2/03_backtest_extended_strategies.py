from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml
from joblib import load


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def annualized_return(nav_end: float, periods: int, periods_per_year: float) -> float:
    if periods <= 0 or nav_end <= 0:
        return float("nan")
    return nav_end ** (periods_per_year / periods) - 1


def max_drawdown(nav: pd.Series) -> float:
    dd = nav / nav.cummax() - 1
    return abs(float(dd.min()))


def sharpe_ratio(ret: pd.Series, periods_per_year: float) -> float:
    std = ret.std()
    if std == 0 or pd.isna(std):
        return float("nan")
    return float(ret.mean() / std * math.sqrt(periods_per_year))


def performance(period_ret: pd.Series, nav: pd.Series, periods_per_year: float) -> dict:
    return {
        "total_return": float(nav.iloc[-1] - 1),
        "annual_return": float(annualized_return(float(nav.iloc[-1]), len(nav), periods_per_year)),
        "volatility": float(period_ret.std() * math.sqrt(periods_per_year)),
        "sharpe": sharpe_ratio(period_ret, periods_per_year),
        "max_drawdown": max_drawdown(nav),
    }


def turnover_cost(codes_by_date: dict[str, list[str]], cost_rate: float) -> pd.Series:
    rows = []
    prev: dict[str, float] = {}
    for date, codes in codes_by_date.items():
        weight = 1.0 / len(codes) if codes else 0.0
        cur = {code: weight for code in codes}
        all_codes = set(prev) | set(cur)
        turnover = sum(abs(cur.get(code, 0.0) - prev.get(code, 0.0)) for code in all_codes)
        rows.append((date, turnover * cost_rate))
        prev = cur
    return pd.Series(dict(rows), name="cost")


def build_model_scores(model_name: str, model, backtest_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    scored = backtest_df.copy()
    scored["model"] = model_name
    scored["score"] = model.predict_proba(scored[feature_cols])[:, 1]
    return scored.sort_values(["rebalance_date", "score"], ascending=[True, False]).reset_index(drop=True)


def decile_backtest(scored: pd.DataFrame, decile_count: int) -> pd.DataFrame:
    rows = []
    for date, group in scored.groupby("rebalance_date", sort=True):
        group = group.sort_values("score", ascending=True).copy()
        group["decile"] = pd.qcut(group["score"].rank(method="first"), decile_count, labels=False) + 1
        for decile, decile_group in group.groupby("decile"):
            rows.append(
                {
                    "rebalance_date": date,
                    "decile": int(decile),
                    "period_ret": float(decile_group["stock_ret_fwd"].mean()),
                    "benchmark_ret": float(decile_group["bench_ret_fwd"].mean()),
                    "count": int(len(decile_group)),
                }
            )
    decile_ret = pd.DataFrame(rows).sort_values(["decile", "rebalance_date"])
    decile_ret["nav"] = decile_ret.groupby("decile")["period_ret"].transform(lambda s: (1 + s).cumprod())
    decile_ret["excess_nav"] = decile_ret.groupby("decile").apply(
        lambda g: (1 + (g["period_ret"] - g["benchmark_ret"])).cumprod()
    ).reset_index(level=0, drop=True)
    return decile_ret


def strategy_backtest(scored: pd.DataFrame, top_n: int, cost_rate: float, index_short_cost_rate: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    dates = sorted(scored["rebalance_date"].astype(str).unique().tolist())
    long_codes: dict[str, list[str]] = {}
    short_codes: dict[str, list[str]] = {}

    for date in dates:
        group = scored[scored["rebalance_date"].astype(str) == date].sort_values("score", ascending=False)
        long_group = group.head(top_n)
        short_group = group.tail(top_n)
        long_codes[date] = long_group["ts_code"].tolist()
        short_codes[date] = short_group["ts_code"].tolist()
        rows.append(
            {
                "rebalance_date": date,
                "next_rebalance_date": str(long_group["next_rebalance_date"].iloc[0]),
                "long_ret": float(long_group["stock_ret_fwd"].mean()),
                "short_bottom_ret": float(short_group["stock_ret_fwd"].mean()),
                "benchmark_ret": float(long_group["bench_ret_fwd"].mean()),
                "long_count": int(len(long_group)),
                "short_count": int(len(short_group)),
            }
        )

    period = pd.DataFrame(rows).sort_values("rebalance_date")
    long_cost = turnover_cost(long_codes, cost_rate).rename("long_cost")
    short_cost = turnover_cost(short_codes, cost_rate).rename("short_cost")
    period = period.merge(long_cost, left_on="rebalance_date", right_index=True, how="left")
    period = period.merge(short_cost, left_on="rebalance_date", right_index=True, how="left")
    period["long_cost"] = period["long_cost"].fillna(0.0)
    period["short_cost"] = period["short_cost"].fillna(0.0)
    period["index_short_cost"] = index_short_cost_rate

    strategy_defs = {
        "long_top10": period["long_ret"] - period["long_cost"],
        "long_top10_short_bottom10": period["long_ret"] - period["short_bottom_ret"] - period["long_cost"] - period["short_cost"],
        "long_top10_short_index": period["long_ret"] - period["benchmark_ret"] - period["long_cost"] - period["index_short_cost"],
    }
    nav_rows = []
    for strategy_name, ret in strategy_defs.items():
        nav = (1 + ret).cumprod()
        tmp = period[["rebalance_date", "next_rebalance_date"]].copy()
        tmp["strategy"] = strategy_name
        tmp["period_ret"] = ret
        tmp["nav"] = nav
        tmp["drawdown"] = nav / nav.cummax() - 1
        nav_rows.append(tmp)
    return pd.concat(nav_rows, ignore_index=True), period


def main() -> None:
    parser = argparse.ArgumentParser(description="十分组、Long-Short、Long-Index 扩展回测。")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--upgrade-config", default=str(DEFAULT_PROJECT_ROOT / "config" / "config_improved.yaml"))
    args = parser.parse_args()

    project_root = Path(args.project_root)
    sys.path.insert(0, str(project_root / "src"))

    from utils.common import read_dataframe, save_dataframe

    base_config = load_yaml(project_root / "config" / "config.yaml")
    upgrade_config = load_yaml(Path(args.upgrade_config))
    processed_dir = project_root / base_config["paths"]["processed"]
    output_dir = project_root / upgrade_config["outputs"]["subdir"]
    report_dir = project_root / upgrade_config["outputs"]["reports_subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_json(output_dir / "dataset_metadata_improved.json")
    feature_cols = metadata["feature_columns"]
    backtest_df = read_dataframe(processed_dir / "backtest_dataset_improved.parquet", dtype={"rebalance_date": str, "ts_code": str})
    top_n = int(upgrade_config["model"]["top_n"])
    decile_count = int(upgrade_config["model"]["decile_count"])
    cost_rate = float(upgrade_config["backtest"]["transaction_cost_bps"]) / 10000.0
    index_short_cost_rate = float(upgrade_config["backtest"].get("index_short_cost_bps", 0)) / 10000.0
    periods_per_year = 252 / base_config["data"]["rebalance_window"]

    model_paths = {
        "random_forest": output_dir / "random_forest_validation_selected.joblib",
        "xgboost": output_dir / "xgboost_validation_selected.joblib",
        "lightgbm": output_dir / "lightgbm_validation_selected.joblib",
    }
    summary_rows = []
    all_nav = []
    all_period = []
    all_decile = []
    all_scores = []

    for model_name, model_path in model_paths.items():
        print(f"[INFO] Backtesting {model_name}")
        model = load(model_path)
        scored = build_model_scores(model_name, model, backtest_df, feature_cols)
        decile = decile_backtest(scored, decile_count)
        decile["model"] = model_name
        nav, period = strategy_backtest(scored, top_n, cost_rate, index_short_cost_rate)
        nav["model"] = model_name
        period["model"] = model_name
        scored_export = scored[["rebalance_date", "next_rebalance_date", "ts_code", "score", "stock_ret_fwd", "bench_ret_fwd", "model"]]

        for strategy_name, group in nav.groupby("strategy"):
            perf = performance(group["period_ret"], group["nav"], periods_per_year)
            summary_rows.append({"model": model_name, "strategy": strategy_name, **perf})

        all_nav.append(nav)
        all_period.append(period)
        all_decile.append(decile)
        all_scores.append(scored_export)

    summary = pd.DataFrame(summary_rows).sort_values(["model", "strategy"])
    save_dataframe(pd.concat(all_nav, ignore_index=True), output_dir / "extended_strategy_nav.csv")
    save_dataframe(pd.concat(all_period, ignore_index=True), output_dir / "extended_strategy_period_returns.csv")
    save_dataframe(pd.concat(all_decile, ignore_index=True), output_dir / "decile_backtest.csv")
    save_dataframe(pd.concat(all_scores, ignore_index=True), output_dir / "improved_scored_panel.parquet")
    save_dataframe(summary, output_dir / "extended_strategy_summary.csv")
    write_json(output_dir / "extended_strategy_summary.json", {"rows": summary.to_dict(orient="records")})

    with (report_dir / "改进版策略回测结果说明.md").open("w", encoding="utf-8") as f:
        f.write("# 改进版策略回测结果说明\n\n")
        f.write("本文件由 `03_backtest_extended_strategies.py` 自动生成，包含三类策略：Long 前10、Long 前10 / Short 后10、Long 前10 / Short 中证500。\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n")
    print("[INFO] 扩展回测完成。")


if __name__ == "__main__":
    main()
