"""Validate published aggregate evidence; this does NOT rerun the research."""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT / 'results' / name).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    names = {'Logistic Regression', 'Random Forest', 'XGBoost', 'LightGBM'}
    prediction = [r for r in rows('prediction_metrics.csv') if r['split'] == 'test']
    assert {r['model'] for r in prediction} == names
    primary = {r['model']: r for r in rows('portfolio_metrics.csv') if r['strategy'] == 'Long Top 10'}
    assert set(primary) == names
    curves = {}
    for row in rows('daily_nav.csv'):
        if row['strategy'] == 'Long Top 10':
            curves.setdefault(row['model'], []).append(row)
    common = None
    for name in names:
        nav = 1.0
        peak = 1.0
        mdd = 0.0
        dates = []
        for row in curves[name]:
            dates.append(row['date'])
            nav *= 1 + float(row['return'])
            peak = max(peak, nav)
            drawdown = nav / peak - 1
            mdd = max(mdd, -drawdown)
            assert math.isclose(nav, float(row['nav']), rel_tol=1e-9, abs_tol=1e-10)
            assert math.isclose(drawdown, float(row['drawdown']), abs_tol=1e-9)
        assert common is None or dates == common
        common = dates
        result = primary[name]
        assert math.isclose(nav - 1, float(result['total_return']), abs_tol=1e-9)
        assert math.isclose(nav ** (252 / len(dates)) - 1, float(result['annual_return']), abs_tol=1e-9)
        assert math.isclose(mdd, float(result['max_drawdown']), abs_tol=1e-9)
    assert len(rows('factor_ablations.csv')) == 40
    assert len(rows('decile_metrics.csv')) == 40
    print('PASS: saved aggregate metrics, compounding, drawdown and common dates.')
    print('This is an evidence consistency check, NOT fresh training/backtesting.')


if __name__ == '__main__':
    main()
