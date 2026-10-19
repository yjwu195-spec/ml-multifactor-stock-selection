# 同学阅读指南 Classmate Guide

## 中文

### 先分清三个层次

本项目不是直接预测“明天股价多少”。它先计算每只合格股票的特征，再学习哪些特征组合更像未来相对收益较高的一组，最后把评分转成投资组合。特征、标签和组合是三个不同层次：特征只能用当时信息，标签包含未来信息而只能用于训练/事后评价，组合决策不能看未来标签。

### 用一个调仓日理解全过程

在信号日收盘后，查当天之前最近的历史成分，剔除当时不满足报价、历史长度、缺失要求的股票。用截至当天的市场数据和已公开财报计算因子。模型对所有合格股票评分，次日开盘尝试买入最高10只。缺报价的订单留现金，已有不能卖出的持仓继续锁定，不用未来信息另挑“替补”。每个交易日估值并计算净值，到下一次调仓再次执行。

未来这一段相对指数的收益可以在事后算标签，但不能先看收益再决定当初哪些股票有资格被打分。课程版修正了早期版本把标签筛选带入候选股票池的问题。

### 四种模型分别在做什么

| 模型 | 直观理解 | 优点 | 需要警惕 |
|---|---|---|---|
| Logistic Regression | 用线性对数几率组合特征 | 简单、便于检查、提供基准 | 难以直接捕捉复杂非线性和交互 |
| Random Forest | 多棵随机化树综合意见 | 降低单树方差、适合非线性 | 树仍会学习噪声，重要性受相关特征影响 |
| XGBoost | 逐步增加树修正损失 | 灵活、正则化机制较丰富 | 树深和迭代多时易拟合金融噪声 |
| LightGBM | 高效梯度提升树 | 训练效率高、可捕捉复杂结构 | 叶子规模和采样不当可能过拟合 |

不要把这张表理解为模型性能排名；具体表现取决于样本、目标和约束。

### 看指标的正确顺序

先检查数据与划分是否合理，再看独立测试预测结果，然后看净收益、回撤、成本和分时期稳定性。准确率54%不是每天有54%概率赚钱；这里分类只覆盖未来相对收益两端的股票。AUC接近0.55代表区分度有限而非极强。Rank IC衡量同一日期分数与后续相对收益的秩相关。Sharpe衡量平均日收益相对波动，最大回撤是历史峰值到谷值损失。它们回答不同问题，不能用单个数字替代全部结论。

### 建议分工方式

可以按数据时点与因子、机器学习验证与解释、组合账本与报告三个方向互相复核，但正式贡献说明仍按实际情况填写；本次三位成员已确认均等贡献。汇报时一人能讲清楚自己的部分，也应能解释为什么不能用回测收益调参。

### 常见误解

- 不是用目前中证500成分回溯；是有陈旧度限制的历史快照。
- 四阶段不是四份完全不同来源的数据，而是按信号日期切分同一个历史面板。
- 调参以训练期交叉验证为主，验证集辅助，不是用测试/回测收益挑最好。
- 因子消融与置换重要性用于解释，不证明因果，不等于SHAP。
- 对冲图不是已实现融券交易；纯多头是主要投资评价。
- 公共仓库没有完整行情，不能仅凭clone就重跑；但可读报告和运行无数据测试。

## English

### The three distinct layers

Features describe information available at the signal time. Labels describe subsequently realized outcomes and are only used for training or retrospective evaluation. A portfolio maps current model scores into orders. Future labels must never decide which contemporaneous candidates receive a score.

### One rebalance cycle

After the signal close, obtain the latest eligible historical constituent snapshot and apply the contemporaneous eligibility rules. Compute market factors from prior/current bars and accounting factors from released statements. Score every eligible stock, then attempt equal-value Top 10 orders at the next session open. Failed entries leave cash; locked exits remain held. Mark daily values and repeat at the next signal. Transaction costs apply to traded notional, not merely the number of changed names.

### The machine learning concepts

Logistic Regression supplies an interpretable linear log-odds baseline. Random Forest aggregates randomized trees, mainly reducing individual-tree variance. XGBoost and LightGBM sequentially add regularized trees to improve an objective, with different engineering and tree-growth choices. More flexible models can fit useful interactions or merely fit noise; low learning rates, shallow trees and minimum leaf/sample constraints therefore matter.

Temporal cross-validation, purging and train-fitted preprocessing address generalization and leakage. Accuracy/AUC evaluate classification, Rank IC evaluates cross-sectional ordering, and portfolio metrics evaluate concentrated investment outcomes. A model can rank differently on those objectives without any arithmetic contradiction.

### A practical reading order

Read the README design and split table, then the PDF/Word report, `prediction_metrics.csv`, `portfolio_metrics.csv`, the NAV/drawdown figure, factor ablations, importance, cost sensitivity and annual slices. Consult `METHODOLOGY.md` when a precise definition is needed. Read `KNOWN_LIMITATIONS.md` before presenting tradability claims.

### Reproduction versus inspection

Opening reports, formatting saved content or checking CSV consistency does not fit a new model. A fresh experiment requires authorized provider inputs, running `prepare -> models -> backtest -> report`, and auditing the regenerated stock-level evidence. The repository preserves original aggregate evidence, not a license to redistribute the input data.

### How to make a new contribution

Create a branch and state the hypothesis before examining a new evaluation period. Keep the existing protocol/result snapshot for comparison. If you repeatedly use current test/backtest outcomes to choose changes, label those results exploratory and reserve a new untouched future window. Changes to reports should follow computed evidence, never manually invented metrics.
