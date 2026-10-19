from pathlib import Path
import argparse
import json
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='MH6805 Group 12 real-data research pipeline')
    parser.add_argument('--stage', choices=['prepare','models','backtest','report','all'], default='all')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'config/research.json').read_text(encoding='utf-8'))
    with threadpool_limits(limits=cfg['n_jobs']):
        if args.stage in ['prepare','all']:
            from src.data import prepare
            prepare(ROOT,cfg)
        if args.stage in ['models','all']:
            from src.modeling import run_models
            run_models(ROOT,cfg)
        if args.stage in ['backtest','all']:
            from src.backtest import run_backtests
            run_backtests(ROOT,cfg)
        if args.stage in ['report','all']:
            from src.reporting import generate
            generate(ROOT,cfg)


if __name__ == '__main__':
    main()
