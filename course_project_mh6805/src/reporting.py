"""Generate figures and a source-driven English course report from actual computed outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from .modeling import classification, ranking

NAMES = ['Logistic Regression','Random Forest','XGBoost','LightGBM']
COLORS = dict(zip(NAMES,['#777777','#1c5588','#bd7730','#28776b']))
SHORT = dict(zip(NAMES,['LR','RF','XGB','LGBM']))


def save(fig, root, name):
    fig.savefig(root / 'figures' / (name+'.png'), dpi=180, bbox_inches='tight')
    fig.savefig(root / 'figures' / (name+'.pdf'), bbox_inches='tight')
    plt.close(fig)


def generate(root, cfg):
    plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15})
    result = root / 'results'
    pred = pd.read_csv(result / 'prediction_metrics.csv')
    nav = pd.read_csv(result / 'daily_nav.csv', parse_dates=['date'])
    portfolio = pd.read_csv(result / 'portfolio_metrics.csv')
    deciles = pd.read_csv(result / 'decile_metrics.csv')
    ablation = pd.read_csv(result / 'factor_ablations.csv')
    importance = pd.read_csv(result / 'permutation_importance.csv')
    annual = pd.read_csv(result / 'annual_performance.csv')
    costs = pd.read_csv(result / 'cost_sensitivity.csv')
    selected = json.loads((result / 'selected_parameters.json').read_text())
    split = json.loads((result / 'splits.json').read_text())
    data = json.loads((result / 'data_audit.json').read_text())
    scores = pd.read_parquet(result / 'predictions.parquet')
    yearly_predictions=[]
    for (name,split_name,year),group in scores.groupby(['model','split',scores.execution_date.dt.year]):
        labeled=group[group.label.notna()]
        yearly_predictions.append({'model':name,'split':split_name,'year':int(year),**classification(labeled.label,labeled.score),'rank_ic':ranking(group).rank_ic.mean()})
    pd.DataFrame(yearly_predictions).to_csv(result/'prediction_by_year.csv',index=False)
    fig, axes = plt.subplots(2,1,figsize=(10,6.2),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for name in NAMES:
        g=nav[(nav.model==name)&(nav.strategy=='Long Top 10')]
        axes[0].plot(g.date,g.nav,label=name,color=COLORS[name],lw=1.4)
        axes[1].plot(g.date,g.drawdown*100,color=COLORS[name],lw=1.1)
    b=g.benchmark_nav
    axes[0].plot(g.date,b,label='CSI 500',color='#222222',ls='--')
    axes[1].plot(g.date,(b / np.maximum.accumulate(np.r_[1.,b])[1:]-1)*100,color='#222222',ls='--')
    axes[0].set_ylabel('Net asset value'); axes[1].set_ylabel('Drawdown (%)')
    axes[0].legend(ncol=3,fontsize=9); fig.tight_layout(); save(fig,root,'daily_nav_drawdown')
    fig, ax=plt.subplots(figsize=(7,4))
    for name in NAMES:
        g=scores[(scores.model==name)&(scores.split=='test')&scores.label.notna()]
        x,y,_=roc_curve(g.label,g.score)
        auc=pred[(pred.model==name)&(pred.split=='test')].auc.iloc[0]
        ax.plot(x,y,label=f'{name} AUC {auc:.3f}',color=COLORS[name])
    ax.plot([0,1],[0,1],':',color='black'); ax.set(xlabel='False positive rate',ylabel='True positive rate'); ax.legend(fontsize=9)
    fig.tight_layout();save(fig,root,'test_roc')
    fig, ax=plt.subplots(figsize=(8,4))
    for name in NAMES:
        g=deciles[deciles.model==name].sort_values('decile')
        ax.plot(g.decile,g.annual_return*100,marker='o',label=name,color=COLORS[name])
    ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Decile (1 lowest score, 10 highest)',ylabel='Cost-adjusted annual return (%)',xticks=range(1,11));ax.legend(fontsize=9)
    fig.tight_layout();save(fig,root,'decile_returns')
    matrix=[];variants=['fundamentals_only','market_only']+['without_'+x for x in data['factor_families']]
    for variant in variants:
        row=[]
        for name in NAMES:
            frame=ablation[ablation.model==name].set_index('variant')
            row.append(frame.loc[variant,'rank_ic']-frame.loc['all_factors','rank_ic'])
        matrix.append(row)
    matrix=np.asarray(matrix)
    fig, ax=plt.subplots(figsize=(8,4.8));lim=max(abs(matrix.min()),abs(matrix.max()))
    im=ax.imshow(matrix,cmap='RdBu',vmin=-lim,vmax=lim,aspect='auto')
    ax.set_xticks(range(4),[SHORT[x] for x in NAMES]);ax.set_yticks(range(len(variants)),[x.replace('without_','Minus ').replace('_',' ') for x in variants])
    for i in range(len(matrix)):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center',fontsize=9,color='white' if abs(matrix[i,j])>lim*.6 else 'black')
    ax.grid(False);fig.colorbar(im,ax=ax,label='Change in test Rank IC versus all factors');fig.tight_layout();save(fig,root,'factor_ablations')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,name in zip(axes.flat,NAMES):
        g=importance[importance.model==name].nlargest(7,'importance').sort_values('importance')
        ax.barh(g.feature,g.importance,color=COLORS[name]);ax.set_title(name);ax.set_xlabel('Increase in test log loss after permutation')
    fig.tight_layout();save(fig,root,'permutation_importance')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name in NAMES:
        g=costs[costs.model==name].sort_values('cost_bps');axes[0].plot(g.cost_bps,g.annual_return*100,marker='o',label=SHORT[name],color=COLORS[name])
        g=annual[annual.model==name];axes[1].plot(g.year,g.total_return*100,marker='o',label=SHORT[name],color=COLORS[name])
    axes[0].set(xlabel='One-way transaction cost (bps)',ylabel='Annual return (%)');axes[1].set(xlabel='Calendar year (first/last partial)',ylabel='Calendar-slice total return (%)')
    axes[0].legend();axes[1].legend();fig.tight_layout();save(fig,root,'costs_and_subperiods')
    fig,ax=plt.subplots(figsize=(8,4))
    test=pred[pred.split=='test'].set_index('model').loc[NAMES]
    x=np.arange(4);ax.bar(x-.18,test.accuracy*100,.36,label='Accuracy',color='#1c5588');ax.bar(x+.18,test.auc*100,.36,label='ROC AUC x 100',color='#28776b')
    ax.set_xticks(x,[SHORT[n] for n in NAMES]);ax.set(ylim=(48,58),ylabel='Test score (%)');ax.legend();fig.tight_layout();save(fig,root,'prediction_comparison')
    # Block bootstrap operates on date-level Rank IC, not independent stock rows.
    ic=pd.read_csv(result/'rank_ic.csv');rng=np.random.default_rng(cfg['seed']);boot=[]
    for name in NAMES:
        values=ic[ic.model==name].rank_ic.dropna().to_numpy();n=len(values);means=[]
        for _ in range(2000):
            starts=rng.integers(0,n,size=int(np.ceil(n/6)))
            sample=np.concatenate([values[(s+np.arange(6))%n] for s in starts])[:n]
            means.append(sample.mean())
        lo,hi=np.quantile(means,[.025,.975]);boot.append({'model':name,'mean_rank_ic':values.mean(),'block_size_periods':6,'lower_95':lo,'upper_95':hi,'periods':n})
    pd.DataFrame(boot).to_csv(result/'rank_ic_bootstrap.csv',index=False)
    # Human-readable model/family output summary is the common source for report and slides.
    clean_portfolio = portfolio.astype(object).where(pd.notna(portfolio), None)
    summary={'data_rows':data['panel_rows'],'features':len(data['features']),'factor_families':data['factor_families'],'splits':split,'prediction_metrics':pred.to_dict('records'),'portfolios':clean_portfolio.to_dict('records'),'selected_parameters':selected,'contribution':'Wu Yangjie, Zhao Jingjie and Wang Zhaobo contributed equally.','caveat':'Financial announcement dates are respected, but historical revision vintages are not independently certified; execution/shorting assumptions remain simplified.'}
    (result/'research_summary.json').write_text(json.dumps(summary,default=str,indent=2),encoding='utf-8')
    write_report_content(root,cfg,summary,importance,ablation,boot)
    write_html(root)
    print('[REPORT] figures, block bootstrap and report content generated',flush=True)


def write_report_content(root,cfg,summary,importance,ablation,boot):
    pred=pd.DataFrame(summary['prediction_metrics']);pm=pd.DataFrame(summary['portfolios']);test=pred[pred.split=='test'].set_index('model');primary=pm[pm.strategy=='Long Top 10'].set_index('model')
    s=summary['splits']; audit=json.loads((root/'results/data_audit.json').read_text());base=pd.read_csv(root/'results/benchmark_metrics.csv').iloc[0]
    training=pd.read_csv(root/'results/training_diagnostics.csv');train=training[training['sample']=='train_before_refit'].set_index('model')
    rf_ab = ablation[ablation.model == 'Random Forest'].set_index('variant')
    rf_ci = next(item for item in boot if item['model'] == 'Random Forest')
    annual = pd.read_csv(root/'results/annual_performance.csv').set_index(['model','year'])
    sections=[
      {'title':'Research question and real financial data','paragraphs':[
       'Abstract. We compare Logistic Regression, Random Forest, XGBoost and LightGBM for cross-sectional CSI 500 stock selection. We rebuild signals from cached Tushare observations, preserve historical membership and use announcement-based accounting availability. A chronological experiment separates training, validation, testing and later portfolio evaluation. Nonlinear models improve held-out discrimination over the linear baseline, but prediction and portfolio rankings differ. The findings remain conditional on data-vintage and execution assumptions, rather than proving a tradable arbitrage.',
       '1 Introduction. Our question is whether nonlinear interactions between firm characteristics and market signals improve stock ranking over an interpretable linear classifier, and whether statistical gains survive portfolio turnover and transaction costs. We test three hypotheses: fundamentals and market signals provide complementary information; tree ensembles capture useful nonlinearities; and stronger prediction does not necessarily imply higher net investment returns. The project extends earlier internship exploration with a newly rebuilt evaluation pipeline, not a reuse of another module submission.',
       f'2 Dataset. Tushare Pro provides daily OHLC/volume, daily_basic valuation and turnover, adjustment factors, financial indicators, balance sheets, the trading calendar and historical CSI 500 weights. The configured raw window is 4 January 2010 to 30 April 2025. There are {summary["data_rows"]:,} eligible stock-date rows, {summary["features"]} factors and {audit["signal_dates"]} usable rebalance dates. Membership is the latest snapshot dated on or before each signal, no more than 62 calendar days old. All raw files have a SHA-256 inventory.',
       'The panel is unbalanced by design. We require a current positive-volume quotation, 60 observed trading records, a valid 60-record momentum factor and no more than 40% missing features. We do not require a stock to survive to the end of the study. The eligibility filter is determined at signal time, before future returns and labels are known. Financial records older than 550 days become missing rather than being carried indefinitely.'
      ],'table':{'headers':['Split','First signal','Last signal','Dates','Rows'],'rows':[[k.title(),s[k]['signal_start'],s[k]['signal_end'],s[k]['dates'],s[k]['inference_rows']] for k in ['train','validation','test','backtest']]},'caption':'Table 1. Chronological samples before purging unavailable training labels.'},
      {'title':'Information timing and model selection','paragraphs':[
       '3 Timing and targets. Signals use information available at the rebalance close and trade at the next exchange session open. A filing becomes available no earlier than the calendar day after its announcement; for balance sheets we use the later valid announcement/actual-announcement date. Late amendments to older report periods cannot replace a newer-period filing. Statement end dates never act as release dates. This conservative date lag cannot reconstruct vendor revision vintages, which remain an explicit limitation.',
       'The label is defined from the next execution-open to following execution-open excess return relative to CSI 500. The upper 30% receive class 1 and the lower 30% class 0. The middle 40% are excluded only from binary training and classification evaluation. Every eligible stock, including the middle group and stocks without observable future labels, remains in the scoring/portfolio universe. Scores discriminate the two extreme classes; they are not calibrated probabilities of beating the index.',
       '4 Preprocessing and validation. Each estimator uses the same features and dates. One-percent/ninety-nine-percent winsorization, median imputation and standardization are fitted only on the relevant training fold. No full-panel fallback median is used. Entirely missing training columns become explicit constant-zero transformed columns, not invented source observations. Parameters minimize mean log loss in three expanding chronological training folds, with validation log loss as a tie-break. Any training example whose label exit is not strictly earlier than the next evaluation signal is purged.',
       f'Three candidate configurations per model produce 36 training-fold fits, plus validation checks. After freezing parameters, models refit on {s["final_training_rows_after_purge"]:,} purged training-plus-validation extreme-label rows. Neither test metrics nor later backtest Sharpe selects parameters or factor sets. Ablations reuse the full-factor selected parameters; they are explanatory held-out comparisons, not a second feature-selection exercise.',
       'Logistic Regression tests a linear log-odds specification with L2 regularization. Random Forest averages randomized trees; depth and minimum leaf counts limit variance. XGBoost and LightGBM sequentially fit regularized trees with low learning rates. LightGBM row sampling is explicitly activated using subsample_freq=1. All random seeds and the finite search grid are saved; the selected settings are not claimed to be globally optimal.'
      ],'table':{'headers':['Model','Selected setting','Mean CV log loss'],'rows':[[SHORT[n],compact_params(summary['selected_parameters'][n]['selected']['params']),f"{summary['selected_parameters'][n]['selected']['cv_log_loss']:.4f}"] for n in NAMES]},'caption':'Table 2. Settings chosen before inspecting held-out results. Full parameter dictionaries are supplied as JSON.'},
      {'title':'Held out prediction and ranking evidence','paragraphs':[
       '5 Predictive performance. Table 3 evaluates the same extreme-label test observations for every model. Rank IC is the Spearman correlation between scores and forward excess returns across the full eligible cross-section with observable realized returns. Missing return endpoints are excluded from evaluation, not from the ex-ante investment candidate list. Classification metrics apply to a different, extreme-label population and must be interpreted accordingly.',
       f'Random Forest has the strongest test accuracy ({test.loc["Random Forest","accuracy"]:.2%}) and AUC ({test.loc["Random Forest","auc"]:.3f}), compared with Logistic Regression ({test.loc["Logistic Regression","accuracy"]:.2%}; AUC {test.loc["Logistic Regression","auc"]:.3f}). This supports useful nonlinearity, but the gains are modest. Training-only RF accuracy is {train.loc["Random Forest","accuracy"]:.2%}; the reduction on later observations illustrates the need for temporal validation rather than a claim that overfitting has disappeared.',
       f'The ROC curves remain close to the diagonal; discrimination is modest. In the later backtest, RF mean Rank IC is {rf_ci["mean_rank_ic"]:.3f}, with a six-period circular block-bootstrap 95% interval of approximately [{rf_ci["lower_95"]:.3f}, {rf_ci["upper_95"]:.3f}]. Blocks use dates, not independent stock rows, to preserve some temporal dependence. This is exploratory evidence from {rf_ci["periods"]} realized holding periods, not a multiple-comparison-adjusted proof of a persistent effect.'
      ],'table':{'headers':['Model','Accuracy','AUC','Log loss','Mean test Rank IC'],'rows':[[SHORT[n],f'{test.loc[n,"accuracy"]:.2%}',f'{test.loc[n,"auc"]:.3f}',f'{test.loc[n,"log_loss"]:.4f}',f'{test.loc[n,"rank_ic_mean"]:.3f}'] for n in NAMES]},'caption':'Table 3. Independent chronological test metrics. Extreme-label samples only for classification.','figures':[('test_roc',3.0,'Figure 1. Held-out ROC curves use identical observations.')]},
      {'title':'Daily portfolio outcomes after trading costs','paragraphs':[
       '6 Portfolio construction. The primary portfolio holds the ten highest-ranked stocks at equal target value. A daily ledger uses fractional adjusted-price units, carries locked holdings through suspensions, skips unavailable opening orders without substituting a future-selected stock, and pays 10 bps on absolute buy/sell notional. Weight drift affects actual turnover. Newly available capital stays as cash if an order cannot be executed. Final liquidations incur the same cost where executable; the last period may be shorter than 20 sessions.',
       'Daily values mark suspended positions at their last observed open without altering raw data or model features. Known delisted positions receive a conservative zero recovery at the delisting date. Maximum drawdown includes initial capital 1.0 and uses daily open marks. It is not an intraday worst-loss measure. The benchmark is the CSI 500 price index on exactly the same opening-date interval.',
       f'In the later portfolio window, LightGBM records {primary.loc["LightGBM","total_return"]:.2%} total return and {primary.loc["LightGBM","max_drawdown"]:.2%} maximum drawdown; RF records {primary.loc["Random Forest","total_return"]:.2%} and {primary.loc["Random Forest","max_drawdown"]:.2%}. The benchmark returns {base.total_return:.2%}. These new outcomes replace the earlier, label-filtered research figures and should not be compared as though only the model parameters changed.'
      ],'table':{'headers':['Model','Total return','Annual return','Sharpe','Max DD'],'rows':[[SHORT[n],f'{primary.loc[n,"total_return"]:.2%}',f'{primary.loc[n,"annual_return"]:.2%}',f'{primary.loc[n,"sharpe"]:.3f}',f'{primary.loc[n,"max_drawdown"]:.2%}'] for n in NAMES]},'caption':'Table 4. Long Top 10 net of 10 bps one-way costs. Sharpe assumes zero risk-free return and 252 sessions/year.','figures':[('daily_nav_drawdown',3.25,'Figure 2. Common-period daily NAV and aligned drawdown.')]},
      {'title':'Factor ablations and model interpretation','paragraphs':[
       '7 Family-level experiments. We refit each model with all factors, fundamentals only, market factors only, and each of seven leave-one-family-out variants. These 40 fits use exactly the same labeled training rows, inference dates and target policy. Hyperparameters remain frozen. The corresponding test-window cost-adjusted portfolio outcomes are also supplied. This isolates information-set effects without selecting the best variant using the test set.',
       f'For RF, all factors yield test Rank IC {rf_ab.loc["all_factors","rank_ic"]:.3f}, versus {rf_ab.loc["fundamentals_only","rank_ic"]:.3f} for fundamentals alone and {rf_ab.loc["market_only","rank_ic"]:.3f} for market factors alone. Removing trading/risk factors changes it to {rf_ab.loc["without_trading_risk","rank_ic"]:.3f}; removing valuation gives {rf_ab.loc["without_valuation","rank_ic"]:.3f}. The combination helps overall, but not every family adds value in this period. An improvement after removing valuation is sample-specific evidence, not a universal rejection of value signals. Correlated factors can substitute for one another.',
       'Permutation importance measures the increase in held-out log loss after independently shuffling one feature. We use a fixed seed, 1,500 real labeled test rows and three repeats for each feature. Importance is predictive, not causal, and can understate correlated substitute variables. We use this explicit importance analysis rather than claiming SHAP explanations that were not computed.'
      ],'figures':[('factor_ablations',3.1,'Figure 3. Test Rank IC change relative to each model with all factors.')], 'table':{'headers':['Model','Three largest permutation importances'],'rows':[[SHORT[n],', '.join(importance[importance.model==n].nlargest(3,'importance').feature)] for n in NAMES]},'caption':'Table 5. Features ranked by increased test log loss; full values and repeat standard deviations are supplied.'},
      {'title':'Turnover costs and stability over time','paragraphs':[
       f'8 Robustness. All models use the same predeclared one-way costs: 0, 5, 10 and 20 bps, without selecting parameters using these curves. Calendar slices use common opening dates; 2020 and 2025 are partial. LightGBM returns {annual.loc[("LightGBM",2024),"total_return"]:.1%} in 2024 and {annual.loc[("LightGBM",2022),"total_return"]:.1%} in 2022, while RF returns {annual.loc[("Random Forest",2022),"total_return"]:.1%} in 2022. Such differences show why aggregate returns alone cannot establish stability.',
       'Ten score-sorted portfolios include every eligible candidate, and their daily ledgers also deduct transaction costs. Group 1 has the lowest score and group 10 the highest. We examine the direction and smoothness of the decile profile rather than demanding perfect monotonicity or presenting a large long-short loss as the drawdown of every group.',
       'Supplementary hedged portfolios use a clearly normalized 50% long / 50% short daily return-spread convention with 100% gross exposure. One hedge subtracts CSI 500 returns and the other subtracts the bottom-stock sleeve; hedge costs are explicitly charged. These are diagnostic overlays, not executable short-sale or futures strategies. Borrow availability, funding, margin, stock limits and futures basis/roll are not fully represented, so the long-only portfolio remains the primary investable approximation.'
      ],'figures':[('costs_and_subperiods',2.05,'Figure 4. Cost sensitivity and calendar-slice returns, without tuning on the displayed results.'),('decile_returns',1.9,'Figure 5. Net annual returns of all ten score groups.')]},
      {'title':'Discussion limitations and references','paragraphs':[
       '9 Discussion. The test evidence favors tree ensembles over the linear baseline, but the highest classification score does not identify the highest-return Top 10 portfolio. Classification loss averages errors across labeled extremes, whereas a concentrated portfolio depends on the ordering and realized tails of only a few names. Turnover, volatility and regime sensitivity therefore matter alongside AUC. We do not claim that boosting is universally superior to Random Forest.',
       'Information timing controls remove several avoidable sources of leakage: report-period end dates are not release dates; preprocessing uses only training data; training label horizons are purged; and future-return labels never define the inference universe. These corrections materially change the experiment. They do not turn cached vendor data into a certified historical-vintage database.',
       'Limitations. Cached financial statements may contain revisions not identifiable from the retained date fields. Historical constituent snapshots have publication/staleness uncertainty. Early-period financial coverage reduces the eligible universe and is reported rather than hidden. Daily-open execution lacks auction fills, price-limit and lot-size rules, market impact and cash-dividend accounting in actual shares. Suspension marks can temporarily understate risk, zero delisting recovery is conservative, and index price returns differ from a total-return benchmark. The small chronological test sample, finite parameter grid, correlated features and many exploratory comparisons limit statistical certainty.',
       'Future work should validate financial vintages, introduce execution and borrow data, test a frozen walk-forward protocol, and assess capacity and uncertainty before any live deployment. Existing out-of-sample outcomes are research evidence under stated assumptions, not a promise of future returns.',
       'Contribution statement. Wu Yangjie, Zhao Jingjie and Wang Zhaobo contributed equally to this group project.'
      ],'references':[
       'Breiman, L. (2001). Random Forests. Machine Learning 45, 5-32. https://doi.org/10.1023/A:1010933404324',
       'Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. https://arxiv.org/abs/1603.02754',
       'Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 30.',
       'Tushare Pro API documentation. Daily: https://tushare.pro/document/2?doc_id=27 ; daily_basic: https://tushare.pro/document/2?doc_id=32',
       'Financial indicators: https://tushare.pro/document/2?doc_id=79 ; balance sheets: https://tushare.pro/document/2?doc_id=78',
       'Trading calendar: https://tushare.pro/document/2?doc_id=26 ; historical index weights: https://tushare.pro/document/2?doc_id=96 ; adjustment factors: https://tushare.pro/document/2?doc_id=28'
      ]}
    ]
    (root/'results/report_content.json').write_text(json.dumps({'title':'Machine Learning Based Multifactor Stock Selection','authors':'Wu Yangjie, Zhao Jingjie and Wang Zhaobo','course':'MH6805 Machine Learning in Finance   Group 12','sections':sections},indent=2),encoding='utf-8')


def compact_params(params):
    if 'C' in params:
        return 'L2 regularization C=' + str(params['C'])
    text = f"{params['n_estimators']} trees; depth {params['max_depth']}"
    if 'min_samples_leaf' in params:
        text += f"; leaf minimum {params['min_samples_leaf']}"
    else:
        text += f"; learning rate {params['learning_rate']}"
    return text


def write_html(root):
    import base64
    import html
    content=json.loads((root/'results/report_content.json').read_text())
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><title>MH6805 Group 12</title><style>body{max-width:1000px;margin:40px auto;padding:30px;background:#f7f7f2;color:#193047;font:18px Georgia,serif;line-height:1.5}h1,h2{line-height:1.15}table{border-collapse:collapse;width:100%;font-size:15px}td,th{border-bottom:1px solid #ccd4d4;padding:8px;text-align:left}img{width:100%;height:auto}section{margin:50px 0}small{font-size:14px}</style>', '<h1>'+html.escape(content['title'])+'</h1><p>'+html.escape(content['authors'])+'</p>']
    for section in content['sections']:
        parts.append('<section><h2>'+html.escape(section['title'])+'</h2>')
        parts.extend('<p>'+html.escape(x)+'</p>' for x in section['paragraphs'])
        if 'table' in section:
            table=section['table'];parts.append('<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in table['headers'])+'</tr>')
            parts.extend('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in table['rows']);parts.append('</table>')
        for stem,height,caption in section.get('figures',[]):
            encoded=base64.b64encode((root/'figures'/f'{stem}.png').read_bytes()).decode()
            parts.append('<img alt="'+html.escape(caption,quote=True)+'" src="data:image/png;base64,'+encoded+'"><small>'+html.escape(caption)+'</small>')
        parts.extend('<p><small>'+html.escape(x)+'</small></p>' for x in section.get('references',[]));parts.append('</section>')
    parts.append('</html>');(root/'deliverables/Group12_Results.html').write_text('\n'.join(parts),encoding='utf-8')
