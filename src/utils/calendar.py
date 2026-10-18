from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import read_dataframe, save_dataframe


def load_trade_calendar(path: Path) -> pd.DataFrame:
    cal = read_dataframe(path, dtype={"cal_date": str, "pretrade_date": str})
    cal = cal[cal["is_open"] == 1].copy()
    cal = cal.sort_values("cal_date").reset_index(drop=True)
    return cal


def build_rebalance_calendar(cal: pd.DataFrame, start_date: str, end_date: str, step: int) -> pd.DataFrame:
    trading_days = cal[(cal["cal_date"] >= start_date) & (cal["cal_date"] <= end_date)].copy()
    rebalance = trading_days.iloc[::step].copy()
    rebalance = rebalance.rename(columns={"cal_date": "rebalance_date"})
    rebalance["rebalance_seq"] = range(len(rebalance))
    rebalance["next_rebalance_date"] = rebalance["rebalance_date"].shift(-1)
    return rebalance.reset_index(drop=True)


def save_rebalance_calendar(df: pd.DataFrame, path: Path) -> None:
    save_dataframe(df, path)
