"""Daily open-price ledger with signal-independent valuation and explicit trading constraints."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .modeling import ranking


def metrics(returns):
    ret = np.asarray(returns, dtype=float)
    if not len(ret) or not np.isfinite(ret).all():
        raise ValueError('Invalid return series')
    nav = np.cumprod(1 + ret)
    if (nav <= 0).any():
        raise ValueError('Portfolio insolvency; do not continue compounding')
    peak = np.maximum.accumulate(np.r_[1., nav])[1:]
    vol = ret.std(ddof=1) * np.sqrt(252)
    return {'total_return': float(nav[-1]-1), 'annual_return': float(nav[-1]**(252/len(ret))-1), 'volatility': float(vol), 'sharpe': float(ret.mean()*252/vol) if vol else 0., 'max_drawdown': float(-(nav/peak-1).min()), 'days': len(ret)}


def execution_arrays(root):
    data = pd.read_parquet(root / 'data/processed/daily_execution.parquet')
    benchmark = pd.read_parquet(root / 'data/processed/benchmark.parquet')
    prices = data.pivot(index='trade_date', columns='ts_code', values='adjusted_open').reindex(benchmark.index)
    volumes = data.pivot(index='trade_date', columns='ts_code', values='vol').reindex(index=prices.index, columns=prices.columns)
    observed = prices.to_numpy(dtype=float)
    tradable = np.isfinite(observed) & (observed > 0) & (volumes.to_numpy() > 0)
    # Carry last observed quotes only for ledger valuation of locked holdings, never as raw observations/features.
    mark = prices.ffill().fillna(0).to_numpy(dtype=float)
    delist = pd.read_csv(root / 'data/processed/delisting.csv', dtype=str).set_index('ts_code').delist_date
    forced = np.zeros_like(tradable)
    for j, code in enumerate(prices.columns):
        date = pd.to_datetime(delist.get(code), errors='coerce')
        if pd.notna(date):
            at = prices.index >= date
            mark[at, j] = 0.
            forced[at, j] = True
            tradable[at, j] = False
    return {'dates': prices.index, 'codes': prices.columns, 'mark': mark, 'tradable': tradable, 'forced_delist': forced, 'benchmark': benchmark.open.to_numpy(dtype=float)}


def simulate(scored, arrays, cost_bps, cfg, selection='top', decile=None):
    codes = arrays['codes']
    lookup = {c:i for i,c in enumerate(codes)}
    instructions = {}
    for date, g in scored.groupby('execution_date'):
        group = g.sort_values(['score','ts_code'], ascending=[False,True])
        if selection == 'decile':
            buckets = np.minimum(np.arange(len(group))*10//len(group), 9)
            chosen = group.iloc[np.where(buckets == (10-decile))[0]]
        elif selection == 'bottom':
            chosen = group.tail(cfg['top_n'])
        else:
            chosen = group.head(cfg['top_n'])
        instructions[pd.Timestamp(date)] = [lookup[c] for c in chosen.ts_code if c in lookup]
    dates = arrays['dates']
    start = dates.searchsorted(min(instructions))
    last_exit = scored.exit_date.max() if 'exit_date' in scored else pd.NaT
    # Test-only ablation window closes at the next rebalance; final backtest liquidates at last observed open.
    end = dates.searchsorted(last_exit) if pd.notna(last_exit) and last_exit < dates[-1] else len(dates)-1
    shares = np.zeros(len(codes))
    cash = 1.
    previous = 1.
    rate = cost_bps / 10000
    rows = []
    for t in range(start, end+1):
        mark = arrays['mark'][t]
        tradeable = arrays['tradable'][t]
        shares[arrays['forced_delist'][t]] = 0.
        values = shares * mark
        equity = cash + values.sum()
        turnover = 0.
        fee = 0.
        blocked = 0
        if dates[t] in instructions or t == end:
            requested = [] if t == end else instructions[dates[t]]
            locked = ~tradeable & (shares > 0)
            free_equity = max(0., equity - values[locked].sum())
            target = values.copy()
            target[tradeable] = 0.
            if requested:
                slot = free_equity / len(requested)
                for j in requested:
                    if tradeable[j]:
                        target[j] = slot
                    else:
                        blocked += 1
            sell = np.maximum(values-target,0) * tradeable
            sold = sell.sum()
            cash += sold * (1-rate)
            shares[tradeable] -= sell[tradeable] / mark[tradeable]
            fee += sold*rate
            buy = np.maximum(target-shares*mark,0) * tradeable
            buy_total = buy.sum()
            scale = min(1., max(cash,0)/(buy_total*(1+rate))) if buy_total > 0 else 0.
            buy *= scale
            shares[tradeable] += buy[tradeable] / mark[tradeable]
            cash -= buy.sum()*(1+rate)
            fee += buy.sum()*rate
            turnover = (sold+buy.sum())/equity if equity > 0 else 0.
        equity = cash + (shares*mark).sum()
        if equity <= 0 or cash < -1e-8:
            raise ValueError('Invalid self-financing ledger')
        held = shares > 1e-12
        rows.append({'date': dates[t], 'return': equity/previous-1, 'nav': equity, 'turnover': turnover, 'cost_fraction': fee/previous, 'cash_weight': cash/equity, 'locked_count': int((held & ~tradeable).sum()), 'blocked_new_orders': blocked, 'holdings': int(held.sum()), 'benchmark_nav': arrays['benchmark'][t]/arrays['benchmark'][start]})
        previous = equity
    result = pd.DataFrame(rows)
    result['benchmark_return'] = result.benchmark_nav.pct_change(fill_method=None).fillna(0)
    result['drawdown'] = result.nav / np.maximum.accumulate(np.r_[1.,result.nav])[1:] - 1
    return result


def run_backtests(root: Path, cfg):
    print('[BACKTEST] loading daily execution prices', flush=True)
    arrays = execution_arrays(root)
    predictions = pd.read_parquet(root / 'results/predictions.parquet')
    summaries, paths, costs, annual, deciles, decile_paths, rank_frames = [], [], [], [], [], [], []
    for name, all_scores in predictions.groupby('model'):
        print('[BACKTEST] ' + name, flush=True)
        scored = all_scores[all_scores.split == 'backtest'].copy()
        ic = ranking(scored)
        ic['model'] = name
        rank_frames.append(ic)
        main = None
        for cost in cfg['cost_scenarios_bps']:
            path = simulate(scored, arrays, cost, cfg)
            costs.append({'model': name, 'cost_bps': cost, **metrics(path['return'])})
            if cost == cfg['stock_cost_bps']:
                main = path
        main['model'] = name
        main['strategy'] = 'Long Top 10'
        summaries.append({'model': name, 'strategy': 'Long Top 10', **metrics(main['return']), 'mean_rebalance_turnover': main.loc[main.turnover>0,'turnover'].mean(), 'blocked_orders': main.blocked_new_orders.sum()})
        paths.append(main)
        for year, period in main.groupby(main.date.dt.year):
            annual.append({'model': name, 'year': year, **metrics(period['return'])})
        bottom_gross = simulate(scored, arrays, 0, cfg, selection='bottom')
        bottom_net = simulate(scored, arrays, cfg['stock_cost_bps'], cfg, selection='bottom')
        for label, returns in {
            '50 Long Top 10 / 50 Short CSI 500': .5*(main['return']-main.benchmark_return)-.5*cfg['index_hedge_cost_bps']/10000*(main.turnover>0),
            '50 Long Top 10 / 50 Short Bottom 10': .5*(main['return']-bottom_gross['return'])-.5*bottom_net.cost_fraction
        }.items():
            diagnostic = main[['date','benchmark_nav','benchmark_return']].copy()
            diagnostic['return'] = returns
            diagnostic['nav'] = (1+returns).cumprod()
            diagnostic['drawdown'] = diagnostic.nav / np.maximum.accumulate(np.r_[1.,diagnostic.nav])[1:] - 1
            diagnostic['model'] = name
            diagnostic['strategy'] = label
            summaries.append({'model': name, 'strategy': label, **metrics(returns)})
            paths.append(diagnostic)
        for decile in range(1,11):
            path = simulate(scored, arrays, cfg['stock_cost_bps'], cfg, selection='decile', decile=decile)
            deciles.append({'model': name, 'decile': decile, **metrics(path['return'])})
            path['model'], path['decile'] = name, decile
            decile_paths.append(path)
    pd.DataFrame(summaries).to_csv(root / 'results/portfolio_metrics.csv', index=False)
    pd.concat(paths, ignore_index=True).to_csv(root / 'results/daily_nav.csv', index=False)
    pd.DataFrame(costs).to_csv(root / 'results/cost_sensitivity.csv', index=False)
    pd.DataFrame(annual).to_csv(root / 'results/annual_performance.csv', index=False)
    pd.DataFrame(deciles).to_csv(root / 'results/decile_metrics.csv', index=False)
    pd.concat(decile_paths, ignore_index=True).to_csv(root / 'results/decile_daily_nav.csv', index=False)
    pd.concat(rank_frames, ignore_index=True).to_csv(root / 'results/rank_ic.csv', index=False)
    ab = pd.read_parquet(root / 'results/ablation_predictions.parquet')
    full = pd.read_parquet(root / 'data/processed/panel.parquet')[['signal_date','ts_code','exit_date']]
    ab = ab.merge(full, on=['signal_date','ts_code'], how='left', validate='many_to_one')
    ab_results = []
    for (name, variant), scored in ab.groupby(['model','variant']):
        path = simulate(scored, arrays, cfg['stock_cost_bps'], cfg)
        ab_results.append({'model': name, 'variant': variant, **metrics(path['return'])})
    pd.DataFrame(ab_results).to_csv(root / 'results/ablation_portfolios.csv', index=False)
    # Benchmark uses exactly the common daily open window.
    main = paths[0]
    pd.DataFrame([metrics(main.benchmark_return)]).to_csv(root / 'results/benchmark_metrics.csv', index=False)
    print('[BACKTEST] daily portfolios, 40 deciles, costs, annual slices and ablations completed', flush=True)
