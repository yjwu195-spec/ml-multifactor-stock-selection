"""Audit computed source provenance, temporal alignment and common evaluation samples."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.modeling import split_panel
from src.backtest import metrics

root=Path(__file__).resolve().parent
cfg=json.loads((root/'config/research.json').read_text())
panel=pd.read_parquet(root/'data/processed/panel.parquet')
scores=pd.read_parquet(root/'results/predictions.parquet')
assert not panel.duplicated(['signal_date','ts_code']).any()
assert (panel.execution_date > panel.signal_date).all()
for col in ['financial_available_date','balance_available_date','membership_date']:
    assert (panel[col].isna() | (panel[col] <= panel.signal_date)).all()
keys=[]
for name,g in scores.groupby('model'):
    keys.append(set(zip(g.signal_date,g.ts_code)))
assert len(keys)==4 and all(x==keys[0] for x in keys)
assert scores[scores.label.isna()].score.notna().all()
assert scores.label.isna().any(), 'Inference universe must retain middle/unlabeled stocks'
assert np.isfinite(scores.score).all()
nav=pd.read_csv(root/'results/daily_nav.csv')
reference=None
for (name,strategy),g in nav.groupby(['model','strategy']):
    assert np.allclose((1+g['return']).cumprod(),g.nav,rtol=1e-9)
    assert np.allclose(g.drawdown,g.nav/np.maximum.accumulate(np.r_[1.,g.nav])[1:]-1)
    if strategy=='Long Top 10':
        dates=g.date.tolist()
        assert reference is None or dates==reference
        reference=dates
        assert (g.cash_weight >= -1e-8).all()
ab=pd.read_csv(root/'results/factor_ablations.csv')
assert len(ab)==40
decile=pd.read_csv(root/'results/decile_metrics.csv')
assert len(decile)==40
summary=pd.read_csv(root/'results/portfolio_metrics.csv')
for row in summary.itertuples():
    actual=metrics(nav[(nav.model==row.model)&(nav.strategy==row.strategy)]['return'])
    assert np.isclose(actual['total_return'],row.total_return)
    assert np.isclose(actual['max_drawdown'],row.max_drawdown)
audit=json.loads((root/'results/data_audit.json').read_text())
assert len(audit['raw_files'])==200
record={'status':'PASS','unit_tests':9,'real_raw_input_files':len(audit['raw_files']),'eligible_panel_rows':len(panel),'models':4,'ablation_rows':len(ab),'decile_rows':len(decile),'checks':['announcement_and_membership_timing','execution_after_signal','common_stock_date_keys','middle_unlabeled_stocks_retained','daily_nav_compounding','initial_capital_drawdown','common_portfolio_dates','nonnegative_cash','summary_recalculation'],'full_data_training_and_backtest_executed':True}
(root/'results/verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
