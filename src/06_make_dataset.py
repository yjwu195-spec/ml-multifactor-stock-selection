from __future__ import annotations

from math import floor

from joblib import dump

from utils.common import ensure_project_dirs, infer_feature_columns, load_config, print_step, read_dataframe, resolve_path, save_dataframe, save_json
from utils.preprocess import cross_section_fillna, cross_section_winsorize, cross_section_zscore, fit_pca_transform


def main() -> None:
    config = load_config()
    ensure_project_dirs(config)
    feature_dir = resolve_path(config["paths"]["features"])
    label_dir = resolve_path(config["paths"]["labels"])
    processed_dir = resolve_path(config["paths"]["processed"])
    output_dir = resolve_path(config["paths"]["outputs"])

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
    dataset = cross_section_winsorize(dataset, feature_cols, "rebalance_date", config["preprocess"]["winsor_n"])
    if config["preprocess"]["standardize"]:
        dataset = cross_section_zscore(dataset, feature_cols, "rebalance_date")

    unique_dates = sorted(dataset["rebalance_date"].unique().tolist())
    model_end = floor(len(unique_dates) * 0.6)
    train_end = floor(model_end * 0.8)
    train_dates = set(unique_dates[:train_end])
    test_dates = set(unique_dates[train_end:model_end])
    backtest_dates = set(unique_dates[model_end:])

    dataset["split"] = dataset["rebalance_date"].map(
        lambda x: "train" if x in train_dates else "test" if x in test_dates else "backtest"
    )

    train_df = dataset[dataset["split"] == "train"].reset_index(drop=True)
    test_df = dataset[dataset["split"] == "test"].reset_index(drop=True)
    backtest_df = dataset[dataset["split"] == "backtest"].reset_index(drop=True)

    final_feature_cols = feature_cols
    if config["preprocess"]["use_pca"]:
        train_df, test_df, backtest_df, final_feature_cols, pca = fit_pca_transform(
            train_df,
            test_df,
            backtest_df,
            feature_cols,
            variance_ratio=config["preprocess"]["pca_variance_ratio"],
        )
        dump(pca, output_dir / "pca.joblib")

    save_dataframe(dataset, processed_dir / "dataset_master.parquet")
    save_dataframe(train_df, processed_dir / "train_dataset.parquet")
    save_dataframe(test_df, processed_dir / "test_dataset.parquet")
    save_dataframe(backtest_df, processed_dir / "backtest_dataset.parquet")

    save_json(
        {
            "feature_columns": final_feature_cols,
            "raw_feature_columns": feature_cols,
            "train_dates": sorted(train_dates),
            "test_dates": sorted(test_dates),
            "backtest_dates": sorted(backtest_dates),
        },
        output_dir / "dataset_metadata.json",
    )
    print_step("建模数据集已生成。")


if __name__ == "__main__":
    main()
