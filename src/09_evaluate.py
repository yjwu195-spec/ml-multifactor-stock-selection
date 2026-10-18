from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd

from utils.common import ensure_project_dirs, load_config, print_step, read_dataframe, resolve_path, save_dataframe, save_json


def annualized_return(nav_end: float, periods: int, periods_per_year: float) -> float:
    if periods <= 0 or nav_end <= 0:
        return float("nan")
    return nav_end ** (periods_per_year / periods) - 1


def max_drawdown(nav_series: pd.Series) -> float:
    rolling_max = nav_series.cummax()
    drawdown = nav_series / rolling_max - 1
    return abs(drawdown.min())


def sharpe_ratio(period_returns: pd.Series, periods_per_year: float) -> float:
    std = period_returns.std()
    if std == 0 or pd.isna(std):
        return float("nan")
    return period_returns.mean() / std * math.sqrt(periods_per_year)


def main() -> None:
    config = load_config()
    ensure_project_dirs(config)
    output_dir = resolve_path(config["paths"]["outputs"])
    report_dir = resolve_path(config["paths"]["reports"])

    period_returns = read_dataframe(output_dir / "backtest_period_returns.csv", dtype={"rebalance_date": str, "next_rebalance_date": str})
    nav = read_dataframe(output_dir / "backtest_nav.csv", dtype={"rebalance_date": str, "next_rebalance_date": str})
    periods_per_year = 252 / config["data"]["rebalance_window"]

    strategy_ret = period_returns["period_stock_ret_net"]
    bench_ret = period_returns["period_bench_ret"]

    summary = {
        "strategy_total_return": float(nav["strategy_nav"].iloc[-1] - 1),
        "benchmark_total_return": float(nav["benchmark_nav"].iloc[-1] - 1),
        "strategy_annual_return": float(annualized_return(nav["strategy_nav"].iloc[-1], len(nav), periods_per_year)),
        "benchmark_annual_return": float(annualized_return(nav["benchmark_nav"].iloc[-1], len(nav), periods_per_year)),
        "strategy_volatility": float(strategy_ret.std() * math.sqrt(periods_per_year)),
        "benchmark_volatility": float(bench_ret.std() * math.sqrt(periods_per_year)),
        "strategy_sharpe": float(sharpe_ratio(strategy_ret, periods_per_year)),
        "benchmark_sharpe": float(sharpe_ratio(bench_ret, periods_per_year)),
        "strategy_max_drawdown": float(max_drawdown(nav["strategy_nav"])),
        "benchmark_max_drawdown": float(max_drawdown(nav["benchmark_nav"])),
    }
    summary["strategy_calmar"] = (
        summary["strategy_annual_return"] / summary["strategy_max_drawdown"]
        if summary["strategy_max_drawdown"] not in [0, float("nan")]
        else float("nan")
    )
    summary["benchmark_calmar"] = (
        summary["benchmark_annual_return"] / summary["benchmark_max_drawdown"]
        if summary["benchmark_max_drawdown"] not in [0, float("nan")]
        else float("nan")
    )

    save_json(summary, output_dir / "performance_summary.json")
    save_dataframe(pd.DataFrame([summary]), output_dir / "performance_summary.csv")

    plt.figure(figsize=(10, 6))
    plt.plot(nav["rebalance_date"], nav["strategy_nav"], label="Strategy")
    plt.plot(nav["rebalance_date"], nav["benchmark_nav"], label="Benchmark")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "nav_comparison.png", dpi=200)
    plt.close()

    with (report_dir / "evaluation_summary.txt").open("w", encoding="utf-8") as f:
        f.write("随机森林多因子选股复刻评估结果\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print_step("评估结果已输出到 outputs 和 reports。")


if __name__ == "__main__":
    main()
