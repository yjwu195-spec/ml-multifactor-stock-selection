"""Single entry point for the portable research project."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(relative: str, *args: str) -> None:
    print(f'[RUN] {relative}', flush=True)
    subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / relative), *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['check', 'download', 'prepare', 'research', 'backtest', 'report', 'all'], default='check')
    args = parser.parse_args()
    if args.stage == 'check':
        run('tools/check_repository.py')
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
    if args.stage in {'download', 'all'}:
        run('src/01_download_tushare.py')
    if args.stage in {'prepare', 'all'}:
        for script in ['00_convert_to_parquet.py', '02_build_calendar.py', '03_build_universe.py', '04_build_features.py', '05_build_labels.py']:
            run('src/' + script)
    if args.stage in {'research', 'all'}:
        for script in ['01_make_dataset_with_validation.py', '02_tune_models_on_validation.py', '03_backtest_extended_strategies.py']:
            run('src/improved_v2/' + script)
    if args.stage == 'backtest':
        run('src/improved_v2/03_backtest_extended_strategies.py')
    if args.stage in {'report', 'research', 'backtest', 'all'}:
        run('tools/build_results_report.py', '--source', 'outputs/improved_v2', '--output', 'outputs/report.html')


if __name__ == '__main__':
    main()
