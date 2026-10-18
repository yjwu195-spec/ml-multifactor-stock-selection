"""Regression checks with tiny deterministic fixtures, never market-data substitutes."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backtest = load_module('extended_backtest', 'src/improved_v2/03_backtest_extended_strategies.py')
dataset = load_module('improved_dataset', 'src/improved_v2/01_make_dataset_with_validation.py')


class ResearchChecks(unittest.TestCase):
    def test_split_dates_are_ordered_and_disjoint(self):
        dates = [f'202001{x:02}' for x in range(1, 21)]
        parts = list(dataset.split_dates(dates, {}).values())
        self.assertEqual([len(x) for x in parts], [10, 2, 2, 6])
        self.assertEqual(set.union(*parts), set(dates))
        for left, right in zip(parts, parts[1:]):
            self.assertLess(max(left), min(right))

    def test_turnover_includes_entry_and_replacement(self):
        costs = backtest.turnover_cost({'a': ['A', 'B'], 'b': ['A', 'B'], 'c': ['C', 'D']}, .001)
        self.assertEqual(costs.tolist(), [.001, 0, .002])

    def test_financial_alignment_uses_no_later_announcement(self):
        from utils.finance import align_financials_to_rebalance, prepare_fina_indicator
        universe = pd.DataFrame({'rebalance_date': ['20200115', '20200215'], 'ts_code': ['A', 'A']})
        raw = pd.DataFrame({'ts_code': ['A', 'A'], 'ann_date': ['20200101', '20200201'], 'end_date': ['20190930', '20191231'], 'roe': [1., 99.]})
        aligned = align_financials_to_rebalance(universe, prepare_fina_indicator(raw), pd.DataFrame())
        self.assertEqual(aligned['roe'].tolist(), [1., 99.])

    def test_reference_summary_matches_nav_endpoints(self):
        nav = pd.read_csv(ROOT / 'results/reference/extended_strategy_nav.csv')
        summary = pd.read_csv(ROOT / 'results/reference/extended_strategy_summary.csv')
        self.assertEqual(len(summary), 9)
        for row in summary.itertuples():
            group = nav[(nav.model == row.model) & (nav.strategy == row.strategy)].sort_values('rebalance_date')
            self.assertAlmostEqual(group.nav.iloc[-1] - 1, row.total_return, places=10)
            self.assertAlmostEqual((1 + group.period_ret).prod(), group.nav.iloc[-1], places=10)

    def test_credential_missing_fails_without_value(self):
        from unittest.mock import patch
        from utils.tushare_client import get_pro_client
        with patch.dict('os.environ', {}, clear=True), patch('dotenv.load_dotenv', return_value=False):
            with self.assertRaisesRegex(ValueError, 'Missing TUSHARE_TOKEN'):
                get_pro_client({'tushare': {'token_env': 'TUSHARE_TOKEN'}})


if __name__ == '__main__':
    unittest.main()
