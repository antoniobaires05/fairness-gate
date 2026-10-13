# Fairness Gates — Project Guide

Senior capstone (CSC 499, Fall 2026). A CI/CD pipeline that audits a credit-approval
ML model for fairness and **fails the build** when a fairness metric crosses its threshold.
The pipeline is the product; the model is intentionally simple.

## Stack
- Python 3.11+, pandas, scikit-learn, Fairlearn, pytest. **No TensorFlow / TFMA / TF fairness libs.**
- GitHub Actions for CI. Later: Azure Blob Storage, Azure SQL (star schema), Tableau.
- Developer is on Windows (PowerShell, `.venv\Scripts\activate`). CI runs on ubuntu-latest.

## Layout
- `ml/ingest.py` — load + clean a dataset into a standard frame: features, `approved` (0/1), protected-attribute columns. All dataset-specific logic lives here.
- `ml/train.py` — train baseline logistic regression, fixed random seed, write `ml/reports/predictions.csv`.
- `ml/fairness_eval.py` — Fairlearn metrics per protected attribute → `ml/reports/fairness_report.json`; exit code 1 if any threshold is violated, printing which metric + attribute failed.
- `ml/fairness_thresholds.yaml` — client-editable thresholds (do not hardcode thresholds in Python).
- `ml/tests/` — unit tests for ingest and metric code (pytest). The gate itself is the exit code, not a pytest test.
- `pipeline/run_gate.js` — Node orchestration/summary (later).
- `data/` — gitignored. Never commit datasets.

## Datasets
1. **Now:** South German Credit (corrected, UCI #573). Target is credit risk (`kredit`: 1 good / 0 bad), used as an approval proxy. Sex is only available mixed into the personal-status field (`famges`) — document this limitation. Age is `alter`; bucket it (e.g. <25 / 25+).
2. **Later:** HMDA (one state, one year). Approval from `action_taken`; drop withdrawn, incomplete, purchased, and preapproval rows. Swapping datasets should only touch `ml/ingest.py`.

## Metrics (per protected attribute)
- `demographic_parity_difference`, `equalized_odds_difference` (Fairlearn) — gated.
- accuracy, precision, recall, selection rate per group — recorded, not gated.
- Report JSON should be one record per metric × group × run (it feeds the warehouse fact table later).

## Conventions
- Small, focused commits with clear messages (`feat:`, `fix:`, `test:`, `ci:`, `docs:`).
- Fixed seeds everywhere (requirement 17: reproducibility). Pin versions in `ml/requirements.txt`.
- No credentials in source — use GitHub Secrets / Azure Key Vault.
- Run `pytest ml/tests` before committing.
