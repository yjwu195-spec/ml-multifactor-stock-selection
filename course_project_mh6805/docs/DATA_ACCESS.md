# 数据获取与输入字典 Data Access and Input Schema

## 中文

### 公开仓库没有哪些数据

`data/raw/`、`data/processed/`、个股级预测Parquet及训练模型不公开。原始缓存约1.5GB量级，不能把付费数据当成代码仓库的普通附件。课程本地交付包含200个缓存文件，其中196个Parquet和4个基础gzip CSV。文件名、大小和SHA-256记录在 `results/data_audit.json`，这里仅发布清单，不发布供应商原始表。

### 怎样取得可运行输入

组员应通过有授权的课程共享渠道取得对应缓存，或用自己具备相应积分权限的Tushare账户重新获取。不要把账号或Token发到Issue、聊天截图、README或代码里。现有课程程序是**缓存驱动**，不会自动调用API，也不会因为设置Token就下载齐数据。私有备份访问需要单独获得权限；本公开仓库不授予该权限，也不自动连接该仓库。

如果自行重新获取，必须按以下文件命名和字段保存；重新查询供应商当前数据可能包含历史修订，不能保证与已发布结果完全一致。首选Parquet，也接受相同文件干名的 `.csv.gz`。`raw_read`逐文件优先读Parquet；同一干名不应故意放置内容不一致的两个版本。压缩缓存是本地兼容格式，不是本次公开上传内容。

### 文件和字段

| 文件干名 | 最少需要的字段 | 解释 |
|---|---|---|
| `trade_cal` | `cal_date,is_open` | 交易所日历；用于固定频率信号和下一交易日 |
| `index_weight_000905_SH` | `trade_date,con_code` | 历史中证500成员快照；不可只取最新成分 |
| `index_daily_000905_SH` | `trade_date,open,close` | 同期指数开盘和收盘价格 |
| `stock_basic` | `ts_code,delist_date` | 包括历史退市信息；不能只含现存股票 |
| `daily_YYYY` | `ts_code,trade_date,open,high,low,close,vol` | 2010至2025逐年行情，2025截至4月30日 |
| `daily_basic_YYYY` | `ts_code,trade_date,turnover_rate,total_mv,pe_ttm,pb,ps_ttm` | 逐年估值、换手、市值 |
| `adj_factor_YYYY` | `ts_code,trade_date,adj_factor` | 逐年实际观察到的复权因子 |
| `fina_indicator_part_*` | 见下方 | 分块财务指标，保留公告和报告期 |
| `balancesheet_part_*` | 见下方 | 分块资产负债表，保留实际公告字段 |

财务指标所需字段：`ts_code,ann_date,end_date,or_yoy,netprofit_yoy,ocf_yoy,roe,roa,grossprofit_margin,netprofit_margin,assets_turn,debt_to_assets,current_ratio,cash_ratio,ocfps`。

资产负债表所需字段：`ts_code,ann_date,f_ann_date,end_date,total_ncl,total_assets,report_type`。

日期建议保留为可解析的 `YYYYMMDD` 字符串，股票代码保留交易所后缀。不能补造缺失行情、用报告期代替公告日、把缺失复权因子一律填1，或把不同百分数单位混入收益小数。真实字段解析和去重规则以 `src/data.py` 为准；`tools/check_inputs.py`只检查文件清单，不替代字段/时点审核。

### 文件放置方式

把授权缓存直接放入本课程目录的 `data/raw/`。不要再套 `raw/raw/`、私有仓库名或日期文件夹。运行 `python tools/check_inputs.py` 检查关键干名和年度范围；它会对比已发布输入哈希，并报告不同版本。哈希不一致不一定意味着数据无效，但说明不是同一个历史缓存，需要记录并重新解释结果。

`prepare`会生成 `data/processed/panel.parquet`、报价、基准和退市表；`models`会生成预测和模型；`backtest`使用这些本地文件。因此只复制公开汇总CSV无法替代完整输入。预留数GB磁盘和足够内存，首次读入全市场年度缓存可能较慢。

### 直接数据源链接

Tushare Pro： [日线行情](https://tushare.pro/document/2?doc_id=27)、[每日指标](https://tushare.pro/document/2?doc_id=32)、[财务指标](https://tushare.pro/document/2?doc_id=79)、[资产负债表](https://tushare.pro/document/2?doc_id=78)、[交易日历](https://tushare.pro/document/2?doc_id=26)、[指数成分权重](https://tushare.pro/document/2?doc_id=96)、[复权因子](https://tushare.pro/document/2?doc_id=28)。实际接口权限和调用限额由供应商账户决定，本仓库不承诺当前账户具有这些权限。

## English

### Access boundary

The public repository intentionally excludes provider observations, stock-level processed panels/predictions and fitted models. The original local run used 200 cached inputs, with an inventory in `results/data_audit.json`. Team members must obtain an authorized private copy or independently reacquire data under their own provider permissions. Access does not imply redistribution rights. No credential is necessary to read reports or test formulas.

### Cache contract

Place the authorized files directly in this course directory's `data/raw/`. The table above gives required stems and minimum fields. Use `.parquet` or the same stem with `.csv.gz`; Parquet takes precedence for each stem. Market files cover 2010 through April 2025. Financial blocks must preserve both release and report-period dates, and balance sheets must retain `f_ann_date`. Historical constituents and delisted-stock information must not be replaced by today's survivors.

Run `python tools/check_inputs.py` before the scientific entry point. The check verifies filenames and compares hashes with the reference inventory; it does not certify schema, provider revision vintages or temporal validity. The full `prepare` stage performs actual field processing and duplicate checks. Different vendor snapshots can produce different outputs even with identical code.

### No automatic download

The course pipeline consumes local caches; it has no Tushare downloader or token loader. Merely setting an environment variable does not acquire data. Do not commit tokens or add an undocumented network fallback. The direct official source links above identify the APIs used; access tiers and quotas must be checked with the provider.

### Reproducibility scope

A public clone is an inspectable source/results release, not a self-contained licensed dataset release. The full local delivery can rerun without an API once its authorized caches are available. Public readers may reproduce report formatting and aggregate checks without raw data, but must not describe those actions as a new empirical run. Missing market/fundamental observations are never replaced with synthetic research inputs.
