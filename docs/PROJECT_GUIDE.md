# Fairness Gates — Project Guide

Senior capstone (CSC 499, Fall 2026). A CI/CD pipeline that audits a credit-approval
ML model for fairness and **fails the build** when a fairness metric crosses its threshold.
The pipeline is the product; the model is intentionally simple.

## Stack
- Python 3.12+ (pinned numpy needs it), pandas, scikit-learn, Fairlearn, pytest. **No TensorFlow / TFMA / TF fairness libs.**
- GitHub Actions for CI. Azure SQL (star schema, `mssql-python` driver), Power BI Desktop for the dashboard (native Azure SQL; Tableau Public cannot connect to it). Later: Azure Blob Storage.
- Developer is on Windows (PowerShell, `.venv\Scripts\activate`). CI runs on ubuntu-latest.

## Layout
- `ml/ingest.py` — the `Dataset` contract (cleaned frame + feature lists + protected columns) and `load_dataset(name)`; South German Credit cleaning.
- `ml/ingest_hmda.py` — HMDA cleaning (row scope, leakage removal, features, protected attributes). Every keep/drop decision is explained in `docs/hmda_data_decisions.md`. All dataset-specific logic lives in the ingest layer.
- `ml/models.py` — common `CreditModel` interface and stand-in vendor models: `vendor_a_logreg` (baseline), `vendor_b_gbm`, `vendor_c_biased` (deliberately biased, for testing the gate), `vendor_a_mitigated` (Fairlearn ThresholdOptimizer).
- `ml/train.py` — train one model on one dataset, fixed seed and split, write `ml/reports/<dataset>/<model>/predictions.csv`.
- `ml/fairness_eval.py` — Fairlearn metrics per protected attribute with 95% confidence intervals → `fairness_report.json` next to the predictions; exit code 1 if any threshold is violated, printing which metric + attribute + groups failed.
- `ml/confidence.py` — Wilson intervals for group rates, stratified bootstrap for gaps.
- `ml/compare.py` — champion vs. challenger: trains every configured model, gates each, recommends keep/promote; exit code = the champion's gate result.
- `ml/fairness_thresholds.yaml` — client-editable thresholds, group-size rules, champion/challenger policy, per dataset (do not hardcode thresholds in Python).
- `ml/tests/` — unit tests for ingest, models, metric and CI code (pytest). The gate itself is the exit code, not a pytest test.
- `warehouse/schema.sql` — star schema: `FactFairnessAudit` (grain: build x dataset x model x group x metric, with CI bounds), `FactModelEvaluation`, dimensions with surrogate keys, Power BI views.
- `warehouse/etl.py` — loads a build's reports: dimensions first (MERGE), then facts replaced per build (no duplicates on re-run). Connection string from `FAIRNESS_SQL_CONNECTION_STRING` (GitHub Secret).
- `docs/azure_setup.md`, `docs/powerbi_dashboard.md` — setup and dashboard build steps.
- `pipeline/run_gate.js` — Node orchestration/summary (later).
- `data/` — gitignored. Never commit datasets.

## Datasets
1. **First:** South German Credit (corrected, UCI #573). Target is credit risk (`kredit`: 1 good / 0 bad), used as an approval proxy. Sex is only available mixed into the personal-status field (`famges`) — document this limitation. Age is `alter`; bucket it (e.g. <25 / 25+).
2. **Now:** HMDA, Delaware 2024 (one state, one year). Approval from `action_taken`; drop withdrawn, incomplete, purchased, and preapproval rows. Swapping datasets should only touch `ml/ingest.py`.

## Metrics (per protected attribute)
- `demographic_parity_difference`, `equalized_odds_difference` (Fairlearn) — gated.
- accuracy, precision, recall, selection rate per group — recorded, not gated.
- Report JSON should be one record per metric × group × run (it feeds the warehouse fact table later).

## Conventions
- Small, focused commits with clear messages (`feat:`, `fix:`, `test:`, `ci:`, `docs:`).
- Fixed seeds everywhere (requirement 17: reproducibility). Pin versions in `ml/requirements.txt`.
- No credentials in source — use GitHub Secrets / Azure Key Vault.
- Run `pytest ml/tests warehouse/tests` before committing.
