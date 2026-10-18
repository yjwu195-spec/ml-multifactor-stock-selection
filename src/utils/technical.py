from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: np.nan})
    return numerator / denominator


def add_adjusted_close(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["adj_close"] = out["close"] * out["adj_factor"]
    return out


def compute_market_factors(stock_df: pd.DataFrame, bench_df: pd.DataFrame, beta_window: int, mfi_window: int) -> pd.DataFrame:
    df = stock_df.copy()
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    df["trade_date"] = df["trade_date"].astype(str)
    bench = bench_df.copy()
    bench["trade_date"] = bench["trade_date"].astype(str)
    bench = bench.sort_values("trade_date")
    bench["bench_ret_1d"] = bench["close"].pct_change()
    df = add_adjusted_close(df)
    df["ret_1d"] = df.groupby("ts_code")["adj_close"].pct_change()
    df["mom20"] = df.groupby("ts_code")["adj_close"].pct_change(20)
    df["mom60"] = df.groupby("ts_code")["adj_close"].pct_change(60)
    df["turnover20"] = df.groupby("ts_code")["turnover_rate"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["turnover60"] = df.groupby("ts_code")["turnover_rate"].transform(lambda s: s.rolling(60, min_periods=20).mean())

    ema12 = df.groupby("ts_code")["adj_close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = df.groupby("ts_code")["adj_close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    dif = ema12 - ema26
    dea = dif.groupby(df["ts_code"]).transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df["macd_close"] = (dif - dea) / df["adj_close"].replace({0: np.nan})

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    money_flow = typical_price * df["vol"].fillna(0)
    tp_diff = typical_price.groupby(df["ts_code"]).diff()
    positive_flow = money_flow.where(tp_diff > 0, 0.0)
    negative_flow = money_flow.where(tp_diff < 0, 0.0).abs()
    pos_sum = positive_flow.groupby(df["ts_code"]).transform(lambda s: s.rolling(mfi_window, min_periods=max(5, mfi_window // 2)).sum())
    neg_sum = negative_flow.groupby(df["ts_code"]).transform(lambda s: s.rolling(mfi_window, min_periods=max(5, mfi_window // 2)).sum())
    money_ratio = _safe_divide(pos_sum, neg_sum)
    df["mfi"] = 100 - (100 / (1 + money_ratio))

    df = df.merge(bench[["trade_date", "bench_ret_1d"]], on="trade_date", how="left")
    beta_frames = []
    for _, group in df.groupby("ts_code", sort=False):
        cov = group["ret_1d"].rolling(beta_window, min_periods=max(20, beta_window // 2)).cov(group["bench_ret_1d"])
        var = group["bench_ret_1d"].rolling(beta_window, min_periods=max(20, beta_window // 2)).var()
        beta_frames.append(_safe_divide(cov, var))
    df["beta"] = pd.concat(beta_frames).sort_index()
    return df
