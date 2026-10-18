from __future__ import annotations

import numpy as np
import pandas as pd


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series(np.nan, index=df.index)


def prepare_fina_indicator(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["ann_date"] = _first_existing(out, ["ann_date", "f_ann_date"]).astype(str)
    out["report_date"] = _first_existing(out, ["end_date", "report_date"]).astype(str)
    out["revenue_yoy"] = _first_existing(out, ["or_yoy", "tr_yoy"])
    out["netprofit_yoy"] = _first_existing(out, ["netprofit_yoy", "profit_yoy", "dt_netprofit_yoy"])
    out["ocf_yoy"] = _first_existing(out, ["ocf_yoy"])
    out["roe"] = _first_existing(out, ["roe", "roe_dt"])
    out["roa"] = _first_existing(out, ["roa"])
    out["grossprofit_margin"] = _first_existing(out, ["grossprofit_margin"])
    out["netprofit_margin"] = _first_existing(out, ["netprofit_margin"])
    out["assets_turn"] = _first_existing(out, ["assets_turn"])
    out["debt_to_assets"] = _first_existing(out, ["debt_to_assets"])
    out["current_ratio"] = _first_existing(out, ["current_ratio"])
    out["cash_ratio"] = _first_existing(out, ["cash_ratio"])
    keep = [
        "ts_code",
        "ann_date",
        "report_date",
        "revenue_yoy",
        "netprofit_yoy",
        "ocf_yoy",
        "roe",
        "roa",
        "grossprofit_margin",
        "netprofit_margin",
        "assets_turn",
        "debt_to_assets",
        "current_ratio",
        "cash_ratio",
    ]
    return out[keep].dropna(subset=["ts_code", "ann_date"]).sort_values(["ts_code", "ann_date"])


def prepare_balancesheet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["ann_date"] = _first_existing(out, ["ann_date", "f_ann_date"]).astype(str)
    out["report_date"] = _first_existing(out, ["end_date", "report_date"]).astype(str)
    total_noncurrent = _first_existing(out, ["total_ncl", "nca_to_assets"])
    total_assets = _first_existing(out, ["total_assets"])
    out["longdebt_to_assets"] = total_noncurrent / total_assets.replace({0: np.nan})
    keep = ["ts_code", "ann_date", "report_date", "longdebt_to_assets"]
    return out[keep].dropna(subset=["ts_code", "ann_date"]).sort_values(["ts_code", "ann_date"])


def align_financials_to_rebalance(universe: pd.DataFrame, fina_df: pd.DataFrame, bs_df: pd.DataFrame) -> pd.DataFrame:
    base = universe[["rebalance_date", "ts_code"]].copy()
    base = base.sort_values(["ts_code", "rebalance_date"])
    base["rebalance_date"] = base["rebalance_date"].astype(str)
    base["rebalance_key"] = pd.to_numeric(base["rebalance_date"], errors="coerce")

    def _merge_asof_values(left: pd.DataFrame, right: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
        out = left.copy()
        if right.empty:
            for col in value_cols:
                if col not in out.columns:
                    out[col] = np.nan
            return out
        merged_parts = []
        for ts_code, group in out.groupby("ts_code", sort=False):
            right_group = right[right["ts_code"] == ts_code].copy()
            if right_group.empty:
                missing = group.copy()
                for col in value_cols:
                    missing[col] = np.nan
                merged_parts.append(missing)
                continue
            right_group["ann_key"] = pd.to_numeric(right_group["ann_date"], errors="coerce")
            group = group.dropna(subset=["rebalance_key"]).copy()
            right_group = right_group.dropna(subset=["ann_key"]).copy()
            if right_group.empty or group.empty:
                missing = group.copy()
                for col in value_cols:
                    missing[col] = np.nan
                merged_parts.append(missing)
                continue
            merged = pd.merge_asof(
                group.sort_values("rebalance_key"),
                right_group[["ann_key"] + value_cols].sort_values("ann_key"),
                left_on="rebalance_key",
                right_on="ann_key",
                direction="backward",
            )
            merged = merged.drop(columns=["ann_key"], errors="ignore")
            merged_parts.append(merged)
        return pd.concat(merged_parts, ignore_index=True)

    fina_value_cols = [
        "revenue_yoy",
        "netprofit_yoy",
        "ocf_yoy",
        "roe",
        "roa",
        "grossprofit_margin",
        "netprofit_margin",
        "assets_turn",
        "debt_to_assets",
        "current_ratio",
        "cash_ratio",
    ]
    bs_value_cols = ["longdebt_to_assets"]

    merged = _merge_asof_values(base, fina_df, fina_value_cols)
    merged = _merge_asof_values(merged, bs_df, bs_value_cols)
    return merged.drop(columns=["rebalance_key"], errors="ignore")
