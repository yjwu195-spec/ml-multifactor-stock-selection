# Publish to a personal GitHub repository

This directory is the publication candidate. Do not upload the old working project wholesale. It contains real provider credentials/data and mixed historical files that are deliberately excluded here.

## Before publication

1. Read `KNOWN_LIMITATIONS.md`; archived performance is affected by future-label-dependent selection.
2. Confirm permission to release internship-related code, Word/PPT content and aggregate research outputs. Provider raw data and broker PDF are not included.
3. Choose repository visibility and an appropriate code license after confirming ownership. No license has been selected on your behalf.
4. Run `python run_research.py --stage check`. This scans common token patterns, not every possible secret type. Review newly added files manually too.

## Git workflow

Create an empty GitHub repository named `ml-multifactor-stock-selection`, then from this directory:

```bash
git init
git add .
git status --short
git commit -m "Prepare CSI 500 machine learning research archive"
git branch -M main
git remote add origin https://github.com/yjwu195-spec/ml-multifactor-stock-selection.git
git push -u origin main
```

These commands have not been executed by the packaging process. If the remote already has a README/commit, reconcile that history instead of force-pushing. Confirm which GitHub account is active before publishing.

`.gitignore` works for Git tracking; browser drag-and-drop does not use it as a security filter. This initial package has no `.env`, real data or model binaries, but after running locally use Git rather than dragging the entire directory into the website. Never use `git add -f` to bypass excluded files. A future accidentally published secret must be revoked; deleting a later file does not remove it from history.

## Included / omitted

Included: executable source, portable config, blank environment example, requirements, docs, tests, aggregate historical tables, Word/PPT, three figures and HTML overview.

Omitted: provider observations, processed panels, full stock-level predictions, serialized models, old backtest-selected search scripts with external paths, Word/PPT QA files, original broker PDF, temporary caches, local credentials and Git history. Obtain provider data through the supplied downloader before executing the complete study.
