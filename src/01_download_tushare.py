from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils.common import ensure_project_dirs, load_config, print_step, save_dataframe, resolve_path
from utils.tushare_client import get_pro_client, pro_call


def month_range(start_date: str, end_date: str) -> list[tuple[str, str]]:
    periods = pd.period_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq="M")
    return [(period.start_time.strftime("%Y%m%d"), period.end_time.strftime("%Y%m%d")) for period in periods]


def trading_dates_from_calendar(cal: pd.DataFrame) -> list[str]:
    open_days = cal[cal["is_open"] == 1].copy()
    return sorted(open_days["cal_date"].astype(str).tolist())


def save_if_missing(df: pd.DataFrame, path: Path, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        print_step(f"跳过已存在文件: {path.name}")
        return False
    save_dataframe(df, path)
    print_step(f"已保存: {path}")
    return True


def download_foundation(pro, config: dict, raw_dir: Path) -> pd.DataFrame:
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]
    benchmark = config["data"]["benchmark_index"]
    overwrite = config["data"]["overwrite_raw"]

    trade_cal = pro_call(
        pro,
        "trade_cal",
        config,
        exchange=config["data"]["exchange"],
        start_date=start_date,
        end_date=end_date,
        fields="exchange,cal_date,is_open,pretrade_date",
    )
    save_if_missing(trade_cal, raw_dir / "trade_cal.csv.gz", overwrite)

    stock_basic_frames = []
    for list_status in ["L", "D", "P"]:
        frame = pro_call(
            pro,
            "stock_basic",
            config,
            exchange="",
            list_status=list_status,
            fields="ts_code,symbol,name,area,industry,market,list_date,delist_date",
        )
        stock_basic_frames.append(frame)
    stock_basic = pd.concat(stock_basic_frames, ignore_index=True).drop_duplicates("ts_code")
    save_if_missing(stock_basic, raw_dir / "stock_basic.csv.gz", overwrite)

    index_daily = pro_call(
        pro,
        "index_daily",
        config,
        ts_code=benchmark,
        start_date=start_date,
        end_date=end_date,
    )
    save_if_missing(index_daily, raw_dir / f"index_daily_{benchmark.replace('.', '_')}.csv.gz", overwrite)

    index_weight_frames = []
    for window_start, window_end in month_range(start_date, end_date):
        frame = pro_call(
            pro,
            "index_weight",
            config,
            index_code=benchmark,
            start_date=window_start,
            end_date=window_end,
        )
        if not frame.empty:
            index_weight_frames.append(frame)
    index_weight = pd.concat(index_weight_frames, ignore_index=True).drop_duplicates()
    save_if_missing(index_weight, raw_dir / f"index_weight_{benchmark.replace('.', '_')}.csv.gz", overwrite)
    return trade_cal


def download_market_panels(pro, config: dict, raw_dir: Path, trading_dates: list[str]) -> None:
    download_market_years(pro, config, raw_dir, years=sorted({date[:4] for date in trading_dates}), trading_dates=trading_dates)


def download_market_years(pro, config: dict, raw_dir: Path, years: list[str], trading_dates: list[str]) -> None:
    overwrite = config["data"]["overwrite_raw"]
    year_to_dates: dict[str, list[str]] = {}
    for trade_date in trading_dates:
        year_to_dates.setdefault(trade_date[:4], []).append(trade_date)

    for year in sorted(years):
        daily_path = raw_dir / f"daily_{year}.csv.gz"
        daily_basic_path = raw_dir / f"daily_basic_{year}.csv.gz"
        adj_path = raw_dir / f"adj_factor_{year}.csv.gz"
        if daily_path.exists() and daily_basic_path.exists() and adj_path.exists() and not overwrite:
            print_step(f"跳过市场面板下载: {year}")
            continue

        year_dates = year_to_dates.get(year, [])
        if not year_dates:
            print_step(f"跳过无交易日年份: {year}")
            continue

        daily_frames = []
        basic_frames = []
        adj_frames = []
        print_step(f"下载 {year} 年日线、daily_basic、adj_factor，共 {len(year_dates)} 个交易日")
        for idx, trade_date in enumerate(year_dates, start=1):
            daily_frames.append(pro_call(pro, "daily", config, trade_date=trade_date))
            basic_frames.append(pro_call(pro, "daily_basic", config, trade_date=trade_date))
            adj_frames.append(pro_call(pro, "adj_factor", config, trade_date=trade_date))
            if idx % 50 == 0 or idx == len(year_dates):
                print_step(f"{year} 年市场数据进度: {idx}/{len(year_dates)}")

        save_if_missing(pd.concat(daily_frames, ignore_index=True), daily_path, overwrite=True)
        save_if_missing(pd.concat(basic_frames, ignore_index=True), daily_basic_path, overwrite=True)
        save_if_missing(pd.concat(adj_frames, ignore_index=True), adj_path, overwrite=True)


def download_financial_panels(pro, config: dict, raw_dir: Path) -> None:
    download_financial_panels_chunked(pro, config, raw_dir, start_part=1, end_part=None)


def download_financial_panels_chunked(
    pro,
    config: dict,
    raw_dir: Path,
    start_part: int = 1,
    end_part: int | None = None,
) -> None:
    overwrite = config["data"]["overwrite_raw"]
    stock_basic = pd.read_csv(raw_dir / "stock_basic.csv.gz", dtype={"ts_code": str})
    ts_codes = stock_basic["ts_code"].dropna().unique().tolist()
    fina_chunk = []
    bs_chunk = []
    chunk_size = 80
    total_parts = (len(ts_codes) + chunk_size - 1) // chunk_size

    print_step(f"下载财务指标与资产负债表数据，共 {total_parts} 个分块。")
    for part_no in range(start_part, total_parts + 1):
        if end_part is not None and part_no > end_part:
            break
        start_idx = (part_no - 1) * chunk_size
        end_idx = min(part_no * chunk_size, len(ts_codes))
        codes_slice = ts_codes[start_idx:end_idx]
        fina_path = raw_dir / f"fina_indicator_part_{part_no:03d}.csv.gz"
        bs_path = raw_dir / f"balancesheet_part_{part_no:03d}.csv.gz"
        if fina_path.exists() and bs_path.exists() and not overwrite:
            print_step(f"跳过已存在财务分块 {part_no:03d}")
            continue

        fina_chunk.clear()
        bs_chunk.clear()
        print_step(f"下载财务分块 {part_no:03d}，股票数 {len(codes_slice)}")
        for idx, ts_code in enumerate(codes_slice, start=1):
            fina_chunk.append(
                pro_call(
                    pro,
                    "fina_indicator",
                    config,
                    ts_code=ts_code,
                    start_date=config["data"]["start_date"],
                    end_date=config["data"]["end_date"],
                )
            )
            bs_chunk.append(
                pro_call(
                    pro,
                    "balancesheet",
                    config,
                    ts_code=ts_code,
                    start_date=config["data"]["start_date"],
                    end_date=config["data"]["end_date"],
                )
            )
            if idx % 20 == 0 or idx == len(codes_slice):
                print_step(f"财务分块 {part_no:03d} 进度: {idx}/{len(codes_slice)}")

        if fina_chunk or bs_chunk:
            fina_path = raw_dir / f"fina_indicator_part_{part_no:03d}.csv.gz"
            bs_path = raw_dir / f"balancesheet_part_{part_no:03d}.csv.gz"
            if overwrite or not fina_path.exists():
                save_dataframe(pd.concat(fina_chunk, ignore_index=True), fina_path)
            if overwrite or not bs_path.exists():
                save_dataframe(pd.concat(bs_chunk, ignore_index=True), bs_path)
            print_step(f"已保存财务分块 {part_no:03d}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Tushare data in smaller, resumable batches.")
    parser.add_argument(
        "--stage",
        choices=["all", "foundation", "market", "financial"],
        default="all",
        help="Choose which download stage to run.",
    )
    parser.add_argument("--year-start", type=int, default=None, help="First year to download for market data.")
    parser.add_argument("--year-end", type=int, default=None, help="Last year to download for market data.")
    parser.add_argument("--financial-start-part", type=int, default=1, help="First financial chunk to download.")
    parser.add_argument("--financial-end-part", type=int, default=None, help="Last financial chunk to download.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    ensure_project_dirs(config)
    raw_dir = resolve_path(config["paths"]["raw"])
    pro = get_pro_client(config)

    trade_cal_path = raw_dir / "trade_cal.csv.gz"
    if args.stage in {"all", "foundation"}:
        print_step("开始下载基础数据")
        trade_cal = download_foundation(pro, config, raw_dir)
    else:
        if not trade_cal_path.exists():
            raise FileNotFoundError("未找到 trade_cal.csv.gz。请先运行 --stage foundation。")
        trade_cal = pd.read_csv(trade_cal_path, dtype={"cal_date": str, "pretrade_date": str})

    if args.stage in {"all", "market"}:
        trading_dates = trading_dates_from_calendar(trade_cal)
        years = sorted({date[:4] for date in trading_dates})
        if args.year_start is not None:
            years = [year for year in years if int(year) >= args.year_start]
        if args.year_end is not None:
            years = [year for year in years if int(year) <= args.year_end]
        print_step(f"开始下载市场面板数据，年份范围: {years[:1]} ... {years[-1:] if years else []}")
        download_market_years(pro, config, raw_dir, years=years, trading_dates=trading_dates)

    if args.stage in {"all", "financial"} and not config["data"]["download_foundation_only"]:
        download_financial_panels_chunked(
            pro,
            config,
            raw_dir,
            start_part=args.financial_start_part,
            end_part=args.financial_end_part,
        )

    print_step("原始数据下载完成")


if __name__ == "__main__":
    main()
