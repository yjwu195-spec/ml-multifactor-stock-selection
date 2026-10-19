# Direct data-source statement

The dataset consists of real cached observations obtained from Tushare Pro. The study uses [daily stock market data](https://tushare.pro/document/2?doc_id=27), [daily basic indicators](https://tushare.pro/document/2?doc_id=32), [financial indicators](https://tushare.pro/document/2?doc_id=79), [balance sheets](https://tushare.pro/document/2?doc_id=78), the [trading calendar](https://tushare.pro/document/2?doc_id=26), [historical index constituents and weights](https://tushare.pro/document/2?doc_id=96), and [adjustment factors](https://tushare.pro/document/2?doc_id=28). The benchmark is CSI 500 (000905.SH).

The raw extraction configuration covers 4 January 2010 to 30 April 2025. Historical membership and actual announcement-date fields are used to construct the panel. Provider access permissions may be required to reacquire the raw tables, but the supplied private/local cache allows the experiments to run without an active API token. Dataset provenance, field definitions and availability limitations are documented in METHODOLOGY.md and data_audit.json.

This paragraph supplies the direct documentation links missing from the originally submitted proposal; the unchanged original proposal is retained as submitted_proposal.pdf for version transparency.
