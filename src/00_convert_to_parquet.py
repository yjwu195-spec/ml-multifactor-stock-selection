from __future__ import annotations

from pathlib import Path

from utils.common import ensure_project_dirs, load_config, print_step, read_dataframe, resolve_path, save_dataframe


def convert_glob(raw_dir: Path, pattern: str) -> int:
    converted = 0
    for csv_path in sorted(raw_dir.glob(pattern)):
        parquet_path = csv_path.with_suffix("").with_suffix(".parquet")
        if parquet_path.exists():
            continue
        print_step(f"转换为 parquet: {csv_path.name}")
        df = read_dataframe(csv_path, dtype={"ts_code": str, "trade_date": str, "ann_date": str, "end_date": str, "f_ann_date": str})
        save_dataframe(df, parquet_path)
        converted += 1
    return converted


def main() -> None:
    config = load_config()
    ensure_project_dirs(config)
    raw_dir = resolve_path(config["paths"]["raw"])
    processed_dir = resolve_path(config["paths"]["processed"])
    feature_dir = resolve_path(config["paths"]["features"])
    label_dir = resolve_path(config["paths"]["labels"])
    output_dir = resolve_path(config["paths"]["outputs"])

    total = 0
    for directory, patterns in [
        (raw_dir, ["daily_*.csv.gz", "daily_basic_*.csv.gz", "adj_factor_*.csv.gz", "fina_indicator_part_*.csv.gz", "balancesheet_part_*.csv.gz"]),
        (processed_dir, ["*.csv.gz"]),
        (feature_dir, ["*.csv.gz"]),
        (label_dir, ["*.csv.gz"]),
        (output_dir, ["*.csv.gz"]),
    ]:
        for pattern in patterns:
            total += convert_glob(directory, pattern)
    print_step(f"parquet 转换完成，共新增 {total} 个文件。")


if __name__ == "__main__":
    main()
