from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config() -> dict:
    config_path = project_root() / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    return project_root() / relative_path


def ensure_project_dirs(config: dict) -> None:
    for rel in config["paths"].values():
        resolve_path(rel).mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    compression = "gzip" if path.suffix.endswith("gz") else None
    df.to_csv(path, index=False, encoding="utf-8-sig", compression=compression)


def read_dataframe(path: Path, **kwargs) -> pd.DataFrame:
    dtype = kwargs.pop("dtype", None)
    if path.suffix == ".parquet":
        columns = kwargs.pop("columns", None)
        df = pd.read_parquet(path, columns=columns, **kwargs)
        if dtype:
            for col, target_dtype in dtype.items():
                if col in df.columns:
                    df[col] = df[col].astype(target_dtype)
        return df
    compression = "gzip" if path.suffix.endswith("gz") else "infer"
    return pd.read_csv(path, compression=compression, dtype=dtype, **kwargs)


def concat_csv_files(paths: Iterable[Path], **kwargs) -> pd.DataFrame:
    frames = [read_dataframe(path, **kwargs) for path in sorted(paths)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def concat_dataframes(paths: Iterable[Path], **kwargs) -> pd.DataFrame:
    frames = [read_dataframe(path, **kwargs) for path in sorted(paths)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def choose_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"未找到候选文件: {[str(path) for path in paths]}")


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_step(message: str) -> None:
    print(f"[INFO] {message}")


KNOWN_NON_FEATURE_COLUMNS = {
    "rebalance_date",
    "next_rebalance_date",
    "ts_code",
    "weight",
    "stock_ret_fwd",
    "bench_ret_fwd",
    "excess_ret",
    "label",
    "split",
}


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols: list[str] = []
    for col in df.columns:
        if col in KNOWN_NON_FEATURE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    return feature_cols
