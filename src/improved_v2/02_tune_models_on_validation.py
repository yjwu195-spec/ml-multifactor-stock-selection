from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import pandas as pd
import yaml
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss


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


def xgboost_grid(random_state: int, n_jobs: int) -> list[dict]:
    common = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    candidates = [
        {"n_estimators": 80, "max_depth": 2, "learning_rate": 0.03, "min_child_weight": 50, "subsample": 0.70, "colsample_bytree": 0.70, "reg_lambda": 10.0, "reg_alpha": 1.0},
        {"n_estimators": 120, "max_depth": 2, "learning_rate": 0.03, "min_child_weight": 30, "subsample": 0.75, "colsample_bytree": 0.75, "reg_lambda": 8.0, "reg_alpha": 0.5},
        {"n_estimators": 160, "max_depth": 2, "learning_rate": 0.02, "min_child_weight": 50, "subsample": 0.80, "colsample_bytree": 0.70, "reg_lambda": 12.0, "reg_alpha": 1.0},
        {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 50, "subsample": 0.70, "colsample_bytree": 0.70, "reg_lambda": 12.0, "reg_alpha": 1.0},
        {"n_estimators": 160, "max_depth": 3, "learning_rate": 0.02, "min_child_weight": 80, "subsample": 0.70, "colsample_bytree": 0.70, "reg_lambda": 15.0, "reg_alpha": 2.0},
        {"n_estimators": 220, "max_depth": 2, "learning_rate": 0.015, "min_child_weight": 80, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 15.0, "reg_alpha": 1.0},
    ]
    return [{**common, **candidate} for candidate in candidates]


def lightgbm_grid(random_state: int, n_jobs: int) -> list[dict]:
    common = {
        "objective": "binary",
        "random_state": random_state,
        "n_jobs": n_jobs,
        "verbose": -1,
        "boosting_type": "gbdt",
    }
    candidates = [
        {"n_estimators": 80, "max_depth": 2, "num_leaves": 4, "learning_rate": 0.03, "min_child_samples": 500, "subsample": 0.70, "colsample_bytree": 0.70, "reg_lambda": 10.0, "reg_alpha": 1.0, "min_split_gain": 0.01},
        {"n_estimators": 120, "max_depth": 2, "num_leaves": 4, "learning_rate": 0.03, "min_child_samples": 300, "subsample": 0.75, "colsample_bytree": 0.75, "reg_lambda": 8.0, "reg_alpha": 0.5, "min_split_gain": 0.01},
        {"n_estimators": 160, "max_depth": 2, "num_leaves": 4, "learning_rate": 0.02, "min_child_samples": 500, "subsample": 0.80, "colsample_bytree": 0.70, "reg_lambda": 12.0, "reg_alpha": 1.0, "min_split_gain": 0.02},
        {"n_estimators": 120, "max_depth": 3, "num_leaves": 7, "learning_rate": 0.03, "min_child_samples": 500, "subsample": 0.70, "colsample_bytree": 0.70, "reg_lambda": 12.0, "reg_alpha": 1.0, "min_split_gain": 0.02},
        {"n_estimators": 160, "max_depth": 3, "num_leaves": 7, "learning_rate": 0.02, "min_child_samples": 800, "subsample": 0.70, "colsample_bytree": 0.70, "reg_lambda": 15.0, "reg_alpha": 2.0, "min_split_gain": 0.03},
        {"n_estimators": 220, "max_depth": 2, "num_leaves": 4, "learning_rate": 0.015, "min_child_samples": 800, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 15.0, "reg_alpha": 1.0, "min_split_gain": 0.02},
    ]
    return [{**common, **candidate} for candidate in candidates]


def score_model(model, x, y) -> dict:
    pred = model.predict(x)
    proba = model.predict_proba(x)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="只使用验证集选择 XGBoost / LightGBM 参数。")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--upgrade-config", default=str(DEFAULT_PROJECT_ROOT / "config" / "config_improved.yaml"))
    args = parser.parse_args()

    project_root = Path(args.project_root)
    sys.path.insert(0, str(project_root / "src"))

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from utils.common import read_dataframe

    base_config = load_yaml(project_root / "config" / "config.yaml")
    upgrade_config = load_yaml(Path(args.upgrade_config))
    processed_dir = project_root / base_config["paths"]["processed"]
    output_dir = project_root / upgrade_config["outputs"]["subdir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_json(output_dir / "dataset_metadata_improved.json")
    feature_cols = metadata["feature_columns"]

    train_df = read_dataframe(processed_dir / "train_dataset_improved.parquet", dtype={"rebalance_date": str, "ts_code": str})
    validation_df = read_dataframe(processed_dir / "validation_dataset_improved.parquet", dtype={"rebalance_date": str, "ts_code": str})
    test_df = read_dataframe(processed_dir / "test_dataset_improved.parquet", dtype={"rebalance_date": str, "ts_code": str})

    x_train, y_train = train_df[feature_cols], train_df["label"].astype(int)
    x_valid, y_valid = validation_df[feature_cols], validation_df["label"].astype(int)
    x_test, y_test = test_df[feature_cols], test_df["label"].astype(int)
    model_cfg = upgrade_config["model"]

    rf = RandomForestClassifier(
        n_estimators=base_config["model"]["n_estimators"],
        max_depth=base_config["model"]["max_depth"],
        min_samples_leaf=base_config["model"]["min_samples_leaf"],
        max_features=base_config["model"]["max_features"],
        random_state=model_cfg["random_state"],
        n_jobs=model_cfg["n_jobs"],
        class_weight=base_config["model"]["class_weight"],
    )
    rf.fit(x_train, y_train)
    rows = []
    baseline = {
        "model": "random_forest",
        "config_id": "baseline",
        "params": json.dumps(rf.get_params(), ensure_ascii=False, sort_keys=True, default=str),
        "train_accuracy": score_model(rf, x_train, y_train)["accuracy"],
        "validation_accuracy": score_model(rf, x_valid, y_valid)["accuracy"],
        "validation_log_loss": score_model(rf, x_valid, y_valid)["log_loss"],
        "test_accuracy": score_model(rf, x_test, y_test)["accuracy"],
        "selected": True,
    }
    rows.append(baseline)
    dump(rf, output_dir / "random_forest_validation_selected.joblib")

    grids = {
        "xgboost": (XGBClassifier, xgboost_grid(model_cfg["random_state"], model_cfg["n_jobs"])),
        "lightgbm": (LGBMClassifier, lightgbm_grid(model_cfg["random_state"], model_cfg["n_jobs"])),
    }
    selected = {"random_forest": baseline}
    selected_params = {"random_forest": baseline["params"]}

    for model_name, (model_cls, grid) in grids.items():
        model_rows = []
        for idx, params in enumerate(grid, start=1):
            print(f"[INFO] Training {model_name} validation config {idx}/{len(grid)}")
            model = model_cls(**params)
            model.fit(x_train, y_train)
            train_score = score_model(model, x_train, y_train)
            valid_score = score_model(model, x_valid, y_valid)
            test_score = score_model(model, x_test, y_test)
            row = {
                "model": model_name,
                "config_id": idx,
                "params": json.dumps(params, ensure_ascii=False, sort_keys=True),
                "train_accuracy": train_score["accuracy"],
                "validation_accuracy": valid_score["accuracy"],
                "validation_log_loss": valid_score["log_loss"],
                "test_accuracy": test_score["accuracy"],
                "selected": False,
            }
            model_rows.append(row)
            rows.append(row)

        # 只按验证集选择参数；若准确率接近，则选择 logloss 更低、训练-验证差距更小的组合。
        best = max(
            model_rows,
            key=lambda r: (
                r["validation_accuracy"],
                -r["validation_log_loss"],
                -abs(r["train_accuracy"] - r["validation_accuracy"]),
            ),
        )
        best["selected"] = True
        selected[model_name] = best
        selected_params[model_name] = best["params"]

        final_params = json.loads(best["params"])
        final_model = model_cls(**final_params)
        final_model.fit(pd.concat([x_train, x_valid]), pd.concat([y_train, y_valid]))
        dump(final_model, output_dir / f"{model_name}_validation_selected.joblib")

    csv_path = output_dir / "validation_tuning_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_json(
        output_dir / "validation_selected_params.json",
        {
            "selection_rule": "validation_accuracy, then lower validation_log_loss, then smaller train_validation_gap",
            "note": "回测集不参与参数选择；最终模型用 train+validation 重新拟合后进入测试和回测。",
            "selected": selected,
        },
    )
    print("[INFO] 验证集调参完成，已保存最优模型。")


if __name__ == "__main__":
    main()
