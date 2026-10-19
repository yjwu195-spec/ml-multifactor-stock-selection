"""Construct features and execution labels from real cached Tushare observations."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

FAMILIES = {
    'valuation': ['earnings_price', 'book_price', 'sales_price', 'ocf_price'],
    'growth': ['revenue_yoy', 'netprofit_yoy', 'ocf_yoy'],
    'quality': ['roe', 'roa', 'grossprofit_margin', 'netprofit_margin', 'assets_turn'],
    'leverage_liquidity': ['debt_to_assets', 'noncurrent_liabilities_assets', 'current_ratio', 'cash_ratio'],
    'momentum': ['momentum20', 'momentum60'],
    'trading_risk': ['turnover20', 'turnover60', 'beta60', 'macd', 'mfi14', 'volatility20'],
    'size': ['log_market_value']
}
FEATURES = sum(FAMILIES.values(), [])


def raw_read(root, stem, columns=None):
    parquet = root / (stem + '.parquet')
    if parquet.exists():
        return pd.read_parquet(parquet, columns=columns)
    return pd.read_csv(root / (stem + '.csv.gz'), usecols=columns, dtype={'ts_code': str, 'trade_date': str, 'cal_date': str})


def statement_parts(root, prefix, columns):
    stems = {p.name.removesuffix('.parquet').removesuffix('.csv.gz') for p in root.glob(prefix + '*') if p.name.endswith(('.parquet','.csv.gz'))}
    if not stems:
        raise FileNotFoundError(f'Missing actual provider statement cache: {prefix}')
    return pd.concat([raw_read(root, stem, columns) for stem in sorted(stems)], ignore_index=True)


def financial_alignment(base, statements, value_cols, max_age):
    """Use only disclosed records, never an earlier-period late revision over a newer filing."""
    parts = []
    for code, group in base.groupby('ts_code', sort=False):
        right = statements[statements.ts_code == code].copy()
        left = group.sort_values('signal_date').copy()
        if right.empty:
            for col in value_cols:
                left[col] = np.nan
            left['available_date'] = pd.NaT
            left['report_end'] = pd.NaT
        else:
            right = right.sort_values(['available_date', 'report_end'])
            right = right[right.report_end >= right.report_end.cummax()]
            right = right.drop_duplicates('available_date', keep='last')
            left = pd.merge_asof(left, right[['available_date', 'report_end'] + value_cols], left_on='signal_date', right_on='available_date', direction='backward')
            stale = (left.signal_date - left.available_date).dt.days > max_age
            left.loc[stale, value_cols] = np.nan
        parts.append(left)
    return pd.concat(parts, ignore_index=True)


def prepare(root: Path, cfg):
    raw = root / 'data/raw'
    out = root / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    cal = raw_read(raw, 'trade_cal')
    days = pd.DatetimeIndex(pd.to_datetime(cal.loc[cal.is_open == 1, 'cal_date'].astype(str))).sort_values().unique()
    days = days[(days >= pd.Timestamp(cfg['start_date'])) & (days <= pd.Timestamp(cfg['end_date']))]
    signals = days[::cfg['rebalance_step']]
    signals = signals[signals + pd.Timedelta(days=1) < days[-1]]
    executions = pd.Series(days[days.get_indexer(signals) + 1], index=signals)
    weights = raw_read(raw, 'index_weight_000905_SH')
    weights['trade_date'] = pd.to_datetime(weights.trade_date.astype(str))
    weights = weights.rename(columns={'con_code': 'ts_code'})
    snapshots = pd.DatetimeIndex(weights.trade_date.unique()).sort_values()
    memberships = []
    for date in signals:
        idx = snapshots.searchsorted(date, side='right') - 1
        if idx < 0:
            continue
        effective = snapshots[idx]
        if (date - effective).days > cfg['maximum_membership_age_days']:
            continue
        group = weights[weights.trade_date == effective][['ts_code']].drop_duplicates().copy()
        group['signal_date'] = date
        group['membership_date'] = effective
        memberships.append(group)
    base = pd.concat(memberships, ignore_index=True)
    codes = sorted(base.ts_code.unique())
    print(f'[DATA] historical membership: {len(codes)} stocks; {len(signals)} signal dates', flush=True)
    market_parts = []
    audit = {'source': 'Tushare cached raw observations', 'raw_files': [], 'duplicate_daily_keys': 0}
    for year in range(days[0].year, days[-1].year + 1):
        print(f'[DATA] reading market year {year}', flush=True)
        daily = raw_read(raw, f'daily_{year}', ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol'])
        daily = daily[daily.ts_code.isin(codes)].copy()
        basic = raw_read(raw, f'daily_basic_{year}', ['ts_code', 'trade_date', 'turnover_rate', 'total_mv', 'pe_ttm', 'pb', 'ps_ttm'])
        basic = basic[basic.ts_code.isin(codes)]
        adj = raw_read(raw, f'adj_factor_{year}', ['ts_code', 'trade_date', 'adj_factor'])
        adj = adj[adj.ts_code.isin(codes)]
        for frame in (daily, basic, adj):
            frame['trade_date'] = pd.to_datetime(frame.trade_date.astype(str))
        audit['duplicate_daily_keys'] += int(daily.duplicated(['ts_code', 'trade_date']).sum())
        if daily.duplicated(['ts_code', 'trade_date']).any() or basic.duplicated(['ts_code', 'trade_date']).any() or adj.duplicated(['ts_code', 'trade_date']).any():
            raise ValueError('Duplicate raw market keys require review')
        merged = daily.merge(basic, on=['ts_code', 'trade_date'], how='left', validate='one_to_one').merge(adj, on=['ts_code', 'trade_date'], how='left', validate='one_to_one')
        for col in merged.select_dtypes('number').columns:
            merged[col] = merged[col].astype('float32')
        market_parts.append(merged)
        del daily, basic, adj, merged
    market = pd.concat(market_parts, ignore_index=True).sort_values(['ts_code', 'trade_date'])
    del market_parts
    market['adjusted_open'] = market.open * market.adj_factor
    market['adjusted_close'] = market.close * market.adj_factor
    market[['ts_code', 'trade_date', 'adjusted_open', 'vol']].to_parquet(out / 'daily_execution.parquet', index=False)
    bench = raw_read(raw, 'index_daily_000905_SH')
    bench['trade_date'] = pd.to_datetime(bench.trade_date.astype(str))
    bench = bench.sort_values('trade_date').set_index('trade_date').reindex(days)
    if bench[['open', 'close']].isna().any().any():
        raise ValueError('Benchmark is incomplete')
    bench[['open', 'close']].to_parquet(out / 'benchmark.parquet')
    br = bench.close.pct_change(fill_method=None)
    factors = []
    for code, group in market.groupby('ts_code', sort=False):
        g = group.set_index('trade_date').sort_index()
        close = g.adjusted_close
        ret = close.pct_change(fill_method=None)
        g['momentum20'] = close.pct_change(20, fill_method=None)
        g['momentum60'] = close.pct_change(60, fill_method=None)
        g['volatility20'] = ret.rolling(20, min_periods=20).std() * np.sqrt(252)
        g['turnover20'] = g.turnover_rate.rolling(20, min_periods=20).mean()
        g['turnover60'] = g.turnover_rate.rolling(60, min_periods=60).mean()
        benchmark_ret = br.reindex(g.index)
        g['beta60'] = ret.rolling(60, min_periods=40).cov(benchmark_ret) / benchmark_ret.rolling(60, min_periods=40).var().replace(0, np.nan)
        diff = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        g['macd'] = (diff - diff.ewm(span=9, adjust=False).mean()) / close
        tp = (g.high + g.low + g.close) / 3
        flow = tp * g.vol
        positive = flow.where(tp.diff() > 0, 0).rolling(14, min_periods=14).sum()
        negative = flow.where(tp.diff() < 0, 0).rolling(14, min_periods=14).sum()
        g['mfi14'] = 100 * positive / (positive + negative).replace(0, np.nan)
        g['earnings_price'] = 1 / g.pe_ttm.replace(0, np.nan)
        g['book_price'] = 1 / g.pb.replace(0, np.nan)
        g['sales_price'] = 1 / g.ps_ttm.replace(0, np.nan)
        g['log_market_value'] = np.log(g.total_mv.where(g.total_mv > 0))
        g['history_count'] = np.arange(1, len(g) + 1)
        cols = sum([v for k, v in FAMILIES.items() if k in ['momentum', 'trading_risk', 'size']], []) + ['earnings_price', 'book_price', 'sales_price', 'close', 'vol', 'history_count']
        selected = g.loc[g.index.isin(signals), cols].copy()
        selected['ts_code'] = code
        selected.index.name = 'signal_date'
        factors.append(selected.reset_index())
    del market
    panel = base.merge(pd.concat(factors, ignore_index=True), on=['signal_date', 'ts_code'], how='left', validate='one_to_one')
    panel['market_eligible'] = (panel.history_count >= cfg['minimum_history']) & (panel.vol > 0) & (panel.close > 0) & panel.momentum60.notna()
    audit['membership_rows'] = len(panel)
    audit['excluded_market_rows'] = int((~panel.market_eligible).sum())
    panel = panel[panel.market_eligible].copy()
    fincols = ['ts_code', 'ann_date', 'end_date', 'or_yoy', 'netprofit_yoy', 'ocf_yoy', 'roe', 'roa', 'grossprofit_margin', 'netprofit_margin', 'assets_turn', 'debt_to_assets', 'current_ratio', 'cash_ratio', 'ocfps']
    fin = statement_parts(raw, 'fina_indicator_part_', fincols)
    fin = fin[fin.ts_code.isin(codes)].rename(columns={'or_yoy': 'revenue_yoy'})
    fin['ann_date'] = pd.to_datetime(fin.ann_date, errors='coerce')
    fin['report_end'] = pd.to_datetime(fin.end_date, errors='coerce')
    fin['available_date'] = fin.ann_date + pd.Timedelta(days=1)
    fin = fin.dropna(subset=['ann_date', 'report_end'])
    fin = fin[fin.ann_date >= fin.report_end]
    audit['financial_duplicate_records'] = int(fin.duplicated(['ts_code', 'ann_date', 'end_date']).sum())
    fin = fin.drop_duplicates(['ts_code', 'ann_date', 'end_date'], keep='first')
    fcols = ['revenue_yoy', 'netprofit_yoy', 'ocf_yoy', 'roe', 'roa', 'grossprofit_margin', 'netprofit_margin', 'assets_turn', 'debt_to_assets', 'current_ratio', 'cash_ratio', 'ocfps']
    panel = financial_alignment(panel, fin, fcols, cfg['maximum_financial_age_days']).rename(columns={'available_date': 'financial_available_date', 'report_end': 'financial_report_end'})
    panel['ocf_price'] = panel.ocfps / panel.close
    bscols = ['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'total_ncl', 'total_assets', 'report_type']
    bs = statement_parts(raw, 'balancesheet_part_', bscols)
    bs = bs[bs.ts_code.isin(codes) & bs.report_type.isin([1, 4, 5])].copy()
    ann = pd.to_datetime(bs.ann_date, errors='coerce')
    actual = pd.to_datetime(bs.f_ann_date, errors='coerce')
    bs['available_date'] = pd.concat([ann, actual], axis=1).max(axis=1) + pd.Timedelta(days=1)
    bs['report_end'] = pd.to_datetime(bs.end_date, errors='coerce')
    bs['noncurrent_liabilities_assets'] = bs.total_ncl / bs.total_assets.replace(0, np.nan)
    bs = bs.dropna(subset=['available_date', 'report_end']).sort_values(['report_type'])
    bs = bs.drop_duplicates(['ts_code', 'available_date', 'report_end'], keep='first')
    panel = financial_alignment(panel, bs, ['noncurrent_liabilities_assets'], cfg['maximum_financial_age_days']).rename(columns={'available_date': 'balance_available_date', 'report_end': 'balance_report_end'})
    panel[FEATURES] = panel[FEATURES].replace([np.inf, -np.inf], np.nan)
    # Eligibility uses only current-date information; no label-dependent filtering here.
    panel['missing_fraction'] = panel[FEATURES].isna().mean(axis=1)
    audit['excluded_missing_feature_rows'] = int((panel.missing_fraction > .4).sum())
    panel = panel[panel.missing_fraction <= .4].copy()
    panel['execution_date'] = panel.signal_date.map(executions)
    next_exec = executions.shift(-1)
    panel['exit_date'] = panel.signal_date.map(next_exec)
    execution = pd.read_parquet(out / 'daily_execution.parquet')
    price = execution[['trade_date', 'ts_code', 'adjusted_open', 'vol']]
    entry = price.rename(columns={'trade_date': 'execution_date', 'adjusted_open': 'entry_open', 'vol': 'entry_volume'})
    exit_price = price.rename(columns={'trade_date': 'exit_date', 'adjusted_open': 'exit_open', 'vol': 'exit_volume'})
    panel = panel.merge(entry, on=['execution_date', 'ts_code'], how='left', validate='many_to_one').merge(exit_price, on=['exit_date', 'ts_code'], how='left', validate='many_to_one')
    valid = (panel.entry_open > 0) & (panel.entry_volume > 0) & (panel.exit_open > 0) & (panel.exit_volume > 0)
    panel['forward_return'] = (panel.exit_open / panel.entry_open - 1).where(valid)
    bopen = bench.open
    panel['benchmark_return'] = panel.exit_date.map(bopen) / panel.execution_date.map(bopen) - 1
    panel['excess_return'] = panel.forward_return - panel.benchmark_return
    panel['label'] = np.nan
    for _, idx in panel.groupby('signal_date').groups.items():
        values = panel.loc[idx, 'excess_return']
        if values.notna().sum() < 30:
            continue
        lo, hi = values.quantile([.3, .7])
        panel.loc[values.index[values <= lo], 'label'] = 0.
        panel.loc[values.index[values >= hi], 'label'] = 1.
    panel = panel.sort_values(['signal_date', 'ts_code']).reset_index(drop=True)
    assert (panel.membership_date <= panel.signal_date).all()
    for col in ['financial_available_date', 'balance_available_date']:
        assert (panel[col].isna() | (panel[col] <= panel.signal_date)).all()
    assert (panel.execution_date > panel.signal_date).all()
    panel.to_parquet(out / 'panel.parquet', index=False)
    basic = raw_read(raw, 'stock_basic')
    basic[['ts_code', 'delist_date']].to_csv(out / 'delisting.csv', index=False)
    audit.update({'panel_rows': len(panel), 'signal_dates': panel.signal_date.nunique(), 'features': FEATURES, 'factor_families': FAMILIES, 'missing_forward_returns': int(panel.forward_return.isna().sum()), 'raw_missing_fraction': panel[FEATURES].isna().mean().to_dict()})
    for path in sorted(raw.iterdir()):
        if path.is_file():
            h = hashlib.sha256()
            with path.open('rb') as handle:
                while block := handle.read(8 * 1024 * 1024):
                    h.update(block)
            audit['raw_files'].append({'file': path.name, 'bytes': path.stat().st_size, 'sha256': h.hexdigest()})
    (root / 'results/data_audit.json').write_text(json.dumps(audit, default=str, indent=2), encoding='utf-8')
    panel.groupby('signal_date').agg(stocks=('ts_code', 'size'), labeled=('label', 'count'), missing_label=('label', lambda x: x.isna().sum())).to_csv(root / 'results/universe_coverage.csv')
    print(f'[DATA] completed: {len(panel)} rows, {len(FEATURES)} factors', flush=True)
