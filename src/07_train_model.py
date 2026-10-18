from __future__ import annotations

from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from utils.common import ensure_project_dirs, load_config, print_step, read_dataframe, read_json, resolve_path, save_json


def build_model(config: dict):
    model_config = config["model"]
    model_type = model_config.get("type", "random_forest").lower()

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=model_config["n_estimators"],
            max_depth=model_config["max_depth"],
            min_samples_leaf=model_config["min_samples_leaf"],
            max_features=model_config["max_features"],
            random_state=model_config["random_state"],
            n_jobs=model_config["n_jobs"],
            class_weight=model_config["class_weight"],
        ), model_type

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("当前 Python 环境未安装 xgboost，请先运行: pip install xgboost") from exc

        xgb_config = model_config.get("xgboost", {})
        return XGBClassifier(
            n_estimators=xgb_config.get("n_estimators", 300),
            max_depth=xgb_config.get("max_depth", 4),
            learning_rate=xgb_config.get("learning_rate", 0.05),
            subsample=xgb_config.get("subsample", 0.8),
            colsample_bytree=xgb_config.get("colsample_bytree", 0.8),
            reg_lambda=xgb_config.get("reg_lambda", 1.0),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method=xgb_config.get("tree_method", "hist"),
            random_state=model_config["random_state"],
            n_jobs=model_config["n_jobs"],
        ), model_type

    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("当前 Python 环境未安装 lightgbm，请先运行: pip install lightgbm") from exc

        lgb_config = model_config.get("lightgbm", {})
        return LGBMClassifier(
            n_estimators=lgb_config.get("n_estimators", 300),
            max_depth=lgb_config.get("max_depth", -1),
            num_leaves=lgb_config.get("num_leaves", 31),
            learning_rate=lgb_config.get("learning_rate", 0.05),
            subsample=lgb_config.get("subsample", 0.8),
            colsample_bytree=lgb_config.get("colsample_bytree", 0.8),
            reg_lambda=lgb_config.get("reg_lambda", 1.0),
            objective="binary",
            random_state=model_config["random_state"],
            n_jobs=model_config["n_jobs"],
            verbose=-1,
        ), model_type

    raise ValueError(f"不支持的模型类型: {model_type}，可选: random_forest, xgboost, lightgbm")


def main() -> None:
    config = load_config()
    ensure_project_dirs(config)
    processed_dir = resolve_path(config["paths"]["processed"])
    output_dir = resolve_path(config["paths"]["outputs"])

    train_df = read_dataframe(processed_dir / "train_dataset.parquet", dtype={"rebalance_date": str, "ts_code": str})
    test_df = read_dataframe(processed_dir / "test_dataset.parquet", dtype={"rebalance_date": str, "ts_code": str})
    metadata = read_json(output_dir / "dataset_metadata.json")
    feature_cols = metadata["feature_columns"]

    model, model_type = build_model(config)

    x_train = train_df[feature_cols]
    y_train = train_df["label"].astype(int)
    x_test = test_df[feature_cols]
    y_test = test_df["label"].astype(int)

    model.fit(x_train, y_train)
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    metrics = {
        "model_type": model_type,
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "classification_report": classification_report(y_test, test_pred, output_dict=True),
        "feature_importance": {
            col: float(value)
            for col, value in zip(feature_cols, getattr(model, "feature_importances_", [0.0] * len(feature_cols)))
        },
    }
    save_json(metrics, output_dir / "model_metrics.json")
    dump(model, output_dir / "model.joblib")
    dump(model, output_dir / f"{model_type}_model.joblib")
    print_step(
        f"{model_type} 模型训练完成，训练集准确率 {metrics['train_accuracy']:.4f}，"
        f"测试集准确率 {metrics['test_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
