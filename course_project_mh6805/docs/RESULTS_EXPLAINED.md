# 课程结果解读 Understanding the Course Results

## 中文

### 一句话结论

本次样本下树模型比逻辑回归显示出一定增量预测价值；随机森林的独立测试指标和纯多头回撤较好，LightGBM的后续回测收益更高。差异不是普适规律，也尚未被证明具有统计显著性。

### 分类指标不等于投资收益

`prediction_metrics.csv` 的 `test` 行对应2019-04至2020-09信号，分类样本是极端两组5,394行，而不是所有股票每天的涨跌。RF正确率54.56%、AUC约0.555，LR正确率52.02%、AUC约0.532。AUC只略高于0.5，说明信息较弱，不能说模型能准确预测市场。

`backtest` 行则是更晚的信号区间，不要把该行的正确率写成独立测试正确率。Rank IC使用可观察未来收益的完整横截面，分类与排序样本定义不完全相同。模型分数也不是已经校准的“跑赢概率”。

### 投资结果如何比较

`portfolio_metrics.csv` 必须先按 `strategy == Long Top 10` 筛选，再比较四个模型。总收益为期末净值相对初始资金的变化；年化收益按252交易日复利；Sharpe以零无风险收益和日收益计算；最大回撤把初始资金1.0纳入峰值，使用每日开盘估值，不等于盘中最大损失。

同一评价期中LightGBM累计80.21%、年化14.72%、回撤19.80%；RF累计60.04%、年化11.59%、回撤17.34%；XGB累计58.47%、年化11.33%、回撤19.20%；LR累计-1.50%。指数累计-7.32%。这些数字已经扣主成本10 bps，但没有完整模拟实盘所有摩擦。

### 为什么不是只选收益最高的模型

RF整体分类略好但LGBM Top 10收益更高，因为后者只依赖评分尾部少数股票及其收益幅度。模型目标函数与集中投资目标并不一致。LightGBM是这一段回测的收益领先者，不是通过独立新样本确认的未来赢家。看完全部回测再选模型是一种新的选择行为，需要额外未见数据验证。

### 十分组和对冲怎么看

每期按分数由低到高分为10组，第1组最低，第10组最高。观察净年化收益是否总体随分数提高，不要求所有相邻组严格单调。十分组每组使用完整每日账本并扣成本。不能把多空收益差的最大回撤误写成某个纯多头分组的回撤。

两类对冲都是50%多头、50%空头收益差，总名义敞口100%，不是100%多头再100%空头。指数对冲可能降低波动与回撤，也会牺牲部分收益。其借券、资金、期货和动态对冲成本未完全实现，因此这里只是诊断性收益序列。

### 消融和重要性说明了什么

40组实验包含每模型的全因子、仅基本面、仅市场类、逐一去掉7个家族。各组使用全因子时选定的参数而非在测试集上再调参。RF全因子测试Rank IC约0.081，基本面单独约0.041，市场单独约0.055；去掉估值后约0.099，这是本样本中的描述性发现，不是可以用测试集直接决定删除估值的理由。

置换重要性在1,500条真实测试标签样本上重复3次，观察打乱单个因子后log loss增加多少。换手类及部分成长、质量因子贡献较突出；相关变量可以互相替代，所以重要性既不是因果系数，也不是唯一的真实经济贡献。此次没有计算SHAP。

### 与旧版和原研报区别

旧探索曾报告XGB累计665.32%，选参使用回测表现，且旧数据流程把未来标签筛选带入选股候选，不能当作本课程独立样本外结果。课程版修正了候选池、财务时点、训练预处理和标签跨界，采用不同分段及每日账本。旧数字下降不能只解释成参数改变。

渤海原研报随机森林测试正确率52.46%、累计收益153.64%、年化17.03%、回撤20.78%，但评价期约6年多，课程版约4年半；数据源和实现也不同。可以讨论方法差异，不能把两个累计收益作同口径竞赛，更不能无证据断言原研报存在某个具体错误。

## English

The tree ensembles show modest predictive gains over the linear baseline in this sample. RF leads the reported independent test classification metrics and has lower primary-portfolio drawdown; LightGBM leads realized later-period return and Sharpe. These are descriptive rankings, not a statistical proof that either model dominates.

Always distinguish `test` and `backtest` predictive rows, and filter portfolio rows by strategy before comparing. Classification evaluates only the extreme labels; ranking uses observable forward returns across the eligible cross-section; Top 10 investment depends on the most highly scored names, return magnitudes, turnover and risk. Accuracy around 54% is not a daily probability of making money.

All headline portfolio figures refer to the common 2020-10-20 to 2025-04-30 daily-open window after 10 bps stock costs. Annualization uses 252 sessions; drawdown includes initial NAV. The benchmark is a price index. The diagnostic hedges use 50/50 return spreads and are not fully executable short-sale/futures books.

Forty factor comparisons and held-out permutation importance help explain the information set. They do not establish causality, compute SHAP, or justify repeatedly optimizing on the same held-out sample. Calendar slices and cost scenarios describe sensitivity rather than select new parameters.

Legacy internship and paper results differ in period, input definitions and experimental controls. In particular, the archived 665.32% XGBoost return is not a valid course headline. Selecting a deployment winner from the now-observed backtests requires a new untouched evaluation period. Remaining vintage, execution, corporate-action and funding limitations are documented explicitly.
