# Packaging verification

Date: 2026-10-19 BJT.

Executed successfully in the available Python environment:

```bash
python run_research.py --stage check
python -m unittest discover -s tests -v
python tools/build_results_report.py
```

Five tests cover chronological split ordering/disjointness, announcement-date financial alignment, turnover entry/replacement formulas, historical summary-vs-NAV consistency, and missing-token failure behavior. Fixtures are deterministic test-only inputs and are not used for performance reporting.

The repository check parses Python source without importing it, detects common 56-character provider token patterns in text and Office XML, checks workstation path literals in Python/YAML and verifies required files. It is not a guarantee against every possible secret format. Word/PPT were copied, not reauthored or visually revalidated in this packaging release.

The HTML renderer was exercised on archived aggregate tables. This verifies display generation only. No provider requests, fresh model fits, full-sample backtest, fresh virtual-environment install, GitHub push or remote Actions run were performed. Source-based research issues remain explicitly documented.

Dependency versions in the available test environment are recorded in `verification_environment.txt`; this records verification context, not the environment of the original historical backtest.
