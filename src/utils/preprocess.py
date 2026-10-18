from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def winsorize_mad(series: pd.Series, n: float = 3.0) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    if pd.isna(mad) or mad == 0:
        return series
    lower = median - n * 1.4826 * mad
    upper = median + n * 1.4826 * mad
    return series.clip(lower=lower, upper=upper)


def cross_section_fillna(df: pd.DataFrame, feature_cols: list[str], date_col: str) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        out[col] = out.groupby(date_col)[col].transform(lambda s: s.fillna(s.median()))
        out[col] = out[col].fillna(out[col].median())
    return out


def cross_section_winsorize(df: pd.DataFrame, feature_cols: list[str], date_col: str, n: float) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        out[col] = out.groupby(date_col)[col].transform(lambda s: winsorize_mad(s, n=n))
    return out


def cross_section_zscore(df: pd.DataFrame, feature_cols: list[str], date_col: str) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        mean = out.groupby(date_col)[col].transform("mean")
        std = out.groupby(date_col)[col].transform("std").replace({0: np.nan})
        out[col] = (out[col] - mean) / std
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def fit_pca_transform(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    backtest_df: pd.DataFrame,
    feature_cols: list[str],
    variance_ratio: float,
):
    pca = PCA(n_components=variance_ratio, svd_solver="full", random_state=42)
    train_components = pca.fit_transform(train_df[feature_cols])
    test_components = pca.transform(test_df[feature_cols])
    backtest_components = pca.transform(backtest_df[feature_cols])
    component_cols = [f"pca_{idx + 1}" for idx in range(train_components.shape[1])]

    def _rebuild(base_df: pd.DataFrame, matrix) -> pd.DataFrame:
        other_cols = [col for col in base_df.columns if col not in feature_cols]
        rebuilt = base_df[other_cols].copy()
        component_df = pd.DataFrame(matrix, columns=component_cols, index=base_df.index)
        return pd.concat([rebuilt, component_df], axis=1)

    return (
        _rebuild(train_df, train_components),
        _rebuild(test_df, test_components),
        _rebuild(backtest_df, backtest_components),
        component_cols,
        pca,
    )
