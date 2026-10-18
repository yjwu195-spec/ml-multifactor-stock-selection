from __future__ import annotations

from utils.calendar import build_rebalance_calendar, load_trade_calendar, save_rebalance_calendar
from utils.common import ensure_project_dirs, load_config, print_step, resolve_path


def main() -> None:
    config = load_config()
    ensure_project_dirs(config)
    raw_dir = resolve_path(config["paths"]["raw"])
    processed_dir = resolve_path(config["paths"]["processed"])

    cal = load_trade_calendar(raw_dir / "trade_cal.csv.gz")
    rebalance = build_rebalance_calendar(
        cal,
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"],
        step=config["data"]["rebalance_window"],
    )
    save_rebalance_calendar(rebalance, processed_dir / "rebalance_calendar.csv")
    print_step(f"已生成调仓日，共 {len(rebalance)} 个")


if __name__ == "__main__":
    main()
