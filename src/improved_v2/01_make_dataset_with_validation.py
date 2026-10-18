from __future__ import annotations

import argparse
import json
import sys
from math import floor
from pathlib import Path

import yaml
from joblib import dump


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_dates(unique_dates: list[str], split_cfg: dict) -> dict[str, set[str]]:
    train_ratio = float(split_cfg.get("train_ratio", 0.50))
    validation_ratio = float(split_cfg.get("validation_ratio", 0.10))
    test_ratio = float(split_cfg.get("test_ratio", 0.10))
    n = len(unique_dates)
    train_end = floor(n * train_ratio)
    validation_end = train_end + floor(n * validation_ratio)
    test_end = validation_end + floor(n * test_ratio)
    return {
        "train": set(unique_dates[:train_end]),
        "validation": set(unique_dates[train_end:validation_end]),
        "test": set(unique_dates[validation_end:test_end]),
        "backtest": set(unique_dates[test_end:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成带验证集的改进版建模数据集。")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--upgrade-config", default=str(DEFAULT_PROJECT_ROOT / "config" / "config_improved.yaml"))
    args = parser.parse_args()

    project_root = Path(args.project_root)
    sys.path.insert(0, str(project_root / "src"))

    from utils.common import infer_feature_columns, read_dataframe, save_dataframe
    from utils.preprocess import cross_section_fillna, cross_section_winsorize, cross_section_zscore, fit_pca_transform

    base_config = load_yaml(project_root / "config" / "config.yaml")
    upgrade_config = load_yaml(Path(args.upgrade_config))

    feature_dir = project_root / base_config["paths"]["features"]
    label_dir = project_root / base_config["paths"]["labels"]
    processed_dir = project_root / base_config["paths"]["processed"]
    output_dir = project_root / upgrade_config["outputs"]["subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    features = read_dataframe(feature_dir / "feature_panel.parquet", dtype={"rebalance_date": str, "ts_code": str})
    labels = read_dataframe(label_dir / "label_panel.parquet", dtype={"rebalance_date": str, "ts_code": str, "next_rebalance_date": str})
    dataset = features.merge(
        labels[["rebalance_date", "next_rebalance_date", "ts_code", "stock_ret_fwd", "bench_ret_fwd", "excess_ret", "label"]],
        on=["rebalance_date", "ts_code"],
        how="inner",
    )
    dataset = dataset[dataset["label"].isin([0, 1])].copy()
    feature_cols = infer_feature_columns(dataset)

    dataset = cross_section_fillna(dataset, feature_cols, "rebalance_date")
    dataset = cross_section_winsorize(dataset, feature_cols, "rebalance_date", base_config["preprocess"]["winsor_n"])
    if base_config["preprocess"]["standardize"]:
        dataset = cross_section_zscore(dataset, feature_cols, "rebalance_date")

    unique_dates = sorted(dataset["rebalance_date"].astype(str).unique().tolist())
    split_map = split_dates(unique_dates, upgrade_config["dataset_split"])

    def label_split(date: str) -> str:
        for split_name, dates in split_map.items():
            if date in dates:
                return split_name
        raise ValueError(f"无法识别日期分组: {date}")

    dataset["split"] = dataset["rebalance_date"].astype(str).map(label_split)
    train_df = dataset[dataset["split"] == "train"].reset_index(drop=True)
    validation_df = dataset[dataset["split"] == "validation"].reset_index(drop=True)
    test_df = dataset[dataset["split"] == "test"].reset_index(drop=True)
    backtest_df = dataset[dataset["split"] == "backtest"].reset_index(drop=True)

    final_feature_cols = feature_cols
    if base_config["preprocess"].get("use_pca", False):
        train_valid = train_df.copy()
        train_df, validation_df, test_df, final_feature_cols, pca = fit_pca_transform(
            train_df,
            validation_df,
            test_df,
            feature_cols,
            variance_ratio=base_config["preprocess"]["pca_variance_ratio"],
        )
        # 复用相同 PCA 转换回测集。
        backtest_features = pca.transform(backtest_df[feature_cols])
        pca_cols = final_feature_cols
        backtest_df = backtest_df.drop(columns=feature_cols).join(
            __import__("pandas").DataFrame(backtest_features, columns=pca_cols, index=backtest_df.index)
        )
        dump(pca, output_dir / "pca_improved.joblib")
        del train_valid

    save_dataframe(dataset, processed_dir / "dataset_master_improved.parquet")
    save_dataframe(train_df, processed_dir / "train_dataset_improved.parquet")
    save_dataframe(validation_df, processed_dir / "validation_dataset_improved.parquet")
    save_dataframe(test_df, processed_dir / "test_dataset_improved.parquet")
    save_dataframe(backtest_df, processed_dir / "backtest_dataset_improved.parquet")

    write_json(
        output_dir / "dataset_metadata_improved.json",
        {
            "feature_columns": final_feature_cols,
            "raw_feature_columns": feature_cols,
            "split_method": "chronological_train_validation_test_backtest",
            "train_dates": sorted(split_map["train"]),
            "validation_dates": sorted(split_map["validation"]),
            "test_dates": sorted(split_map["test"]),
            "backtest_dates": sorted(split_map["backtest"]),
            "row_counts": {
                "train": len(train_df),
                "validation": len(validation_df),
                "test": len(test_df),
                "backtest": len(backtest_df),
            },
        },
    )
    print("[INFO] 已生成带验证集的改进版数据集。")


if __name__ == "__main__":
    main()
