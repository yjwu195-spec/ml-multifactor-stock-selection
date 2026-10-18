"""Build a self-contained report from saved result tables, not a backtest."""
import argparse
import csv
import html
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='results/reference')
    parser.add_argument('--output', default='reports/results_overview.html')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / args.source
    with (source / 'extended_strategy_summary.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    headers = ['model', 'strategy', 'total_return', 'annual_return', 'sharpe', 'max_drawdown']
    table = '<tr>' + ''.join('<th>' + h + '</th>' for h in headers) + '</tr>'
    for row in rows:
        cells = []
        for h in headers:
            value = row[h]
            if h in {'total_return', 'annual_return', 'max_drawdown'}:
                value = f'{float(value):.2%}'
            elif h == 'sharpe':
                value = f'{float(value):.3f}'
            cells.append('<td>' + html.escape(value) + '</td>')
        table += '<tr>' + ''.join(cells) + '</tr>'
    content = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>CSI 500 ML research</title>
<style>body{font:16px Georgia,serif;max-width:1200px;margin:40px auto;padding:24px;background:#f5f3ed;color:#193047}table{border-collapse:collapse;width:100%;font:14px sans-serif}td,th{padding:12px;border-bottom:1px solid #ccd3d9;text-align:left}th{background:#193047;color:white}.note{padding:20px;background:#fff2ce}</style>
<h1>Machine Learning Multifactor Stock Selection</h1>
<p>Random Forest / XGBoost / LightGBM; historical CSI 500 constituents; 20-trading-day rebalancing.</p>
<p class="note">Historical research snapshot. The legacy backtest filters the investment universe by future-return labels. This is an ex-post selection bias: these results are not evidence of implementable out-of-sample performance. See docs/KNOWN_LIMITATIONS.md. This renderer does not run a backtest.</p>
<h2>Archived model and portfolio comparison</h2><table>''' + table + '''</table>
<h2>Methods and provenance</h2><p>Source: Tushare Pro. Sample configuration: 2010-01-04 to 2025-04-30. Chronological 50/10/10/30 split. Boosting parameters selected by validation accuracy, then log loss and train-validation gap. Tables are copied from existing local outputs; no new full-sample run was performed during packaging.</p>
<h2>Portfolio assumptions</h2><p>Long Top 10; Long Top 10 / Short Bottom 10; Long Top 10 / Short CSI 500. Stock transaction cost: 10 bps per absolute weight turnover. Index short cost: 1 bp per rebalance period. Gross exposure of long-short portfolios is approximately 200% on the return-spread convention; financing and borrow constraints are not modeled.</p>
<h2>Validation and remaining work</h2><p>Use a full, label-independent investment universe, fit preprocessing without future information, purge overlapping label horizons, and rerun results. Period-frequency drawdown may understate intraperiod drawdown. Data licensing, point-in-time fundamentals, delisted-stock coverage and execution restrictions need further verification.</p></html>'''
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
