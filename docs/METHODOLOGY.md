# Fairness Gates: methodology and project record

Senior capstone, CSC 499, Fall 2026. Repository: <https://github.com/antoniobaires05/fairness-gate>

This document records **everything built up to 2026-10-16**: what the pipeline does
and how, the models and the math behind them, every significant decision and why it
was made, the errors hit along the way and how they were fixed, and the full commit
history. It's written so someone (or a new Claude Code session) can pick up the
project with no other context.

Companion documents:
- `docs/PROJECT_GUIDE.md`: rules and conventions (loaded by `CLAUDE.md`)
- `docs/hmda_data_decisions.md`: every HMDA row and column keep/drop decision
- `docs/azure_setup.md`: Azure SQL setup
- `docs/powerbi_dashboard.md`: building the dashboard

---

## 1. Status at a glance

| Area | Status |
|---|---|
| Scope | **One dataset (HMDA), three models.** South German Credit and the deliberately biased Vendor C were removed on 2026-10-16 (§16) |
| HMDA (Delaware 2024) ingest | ✅ done, 16,511 decided in-scope applications |
| Model interface + 3 stand-in vendor models | ✅ done |
| Fairness gate with 95% confidence intervals | ✅ done |
| Champion vs. challenger comparison | ✅ done |
| Thresholds and policy in a config file | ✅ done (`ml/fairness_thresholds.yaml`) |
| GitHub Actions CI | ✅ working end to end (run #12 passes, run #13 fails as intended) |
| Azure SQL star schema + ETL | ✅ live on `fairness-gates-ab` / `FairnessGates` |
| Power BI dashboard | ⏳ **next**: Power BI Desktop not yet installed; Phase 1 instructions given |
| Tests | 123 passing, 1 opt-in database test (passes against LocalDB) |

---

## 2. What the project is

A **CI/CD pipeline that audits a credit-approval model for fairness and fails the
build when a fairness metric crosses its threshold.** The pipeline is the product;
the models are intentionally simple stand-ins for third-party "vendor" models a
lender might buy.

On every push or manual run, GitHub Actions:
1. downloads the data,
2. runs the unit tests,
3. cleans HMDA mortgage data,
4. trains three vendor models on the same split,
5. measures each model's fairness across race, ethnicity, sex and age, with confidence intervals,
6. **fails the build if the production ("champion") model breaks a threshold**,
7. recommends whether a challenger should replace the champion,
8. loads the whole audit into an Azure SQL star schema that Power BI reads.

---

## 3. How the pipeline works

```
                    ┌──────────────── GitHub Actions (.github/workflows/fairness-gate.yml) ─────────────────┐
 push / PR /        │                                                                                       │
 "Run workflow" ──▶ │ Fetch datasets ─▶ pytest ─▶ ml.ingest_hmda ─▶ ml.compare ─▶ warehouse.etl ─▶ artifact │
 (champion choice)  │  CFPB HMDA        123 tests   row funnel        trains 3 models   Azure SQL      reports  │
                    │  SHA-256 checks                                 gates each        (even when      zip     │
                    │                                                 exit = champion   gate fails)             │
                    └───────────────────────────────────────────────────────────────────────────────────────┘
                                                                              │
   ml/ingest.py ──▶ Dataset ──▶ ml/train.py ──▶ predictions.csv ──▶ ml/fairness_eval.py ──▶ fairness_report.json
   ml/ingest_hmda.py           ml/models.py                        ml/confidence.py              │
                                                                   fairness_thresholds.yaml       ▼
                                          ml/compare.py ──▶ comparison.json (run block + decision + models)
                                                                              │
                                                                              ▼
                                    warehouse/etl.py ──▶ Azure SQL star schema ──▶ Power BI Desktop
```

### 3.1 Module by module

| File | Role |
|---|---|
| `ml/ingest.py` | Defines the **`Dataset` contract**: cleaned frame, numeric features, categorical features, protected (audit) columns, target `approved`. `load_dataset(name)` returns it (only `hmda` exists). |
| `ml/ingest_hmda.py` | HMDA cleaning: row scope, leakage removal, feature engineering, protected attributes. Run it directly to print the row funnel. |
| `ml/models.py` | `CreditModel` interface (`fit`, `predict_proba`, `predict`) and the three vendor models. |
| `ml/train.py` | Fixed seed (42), stratified 70/30 split, train one model, write `ml/reports/<dataset>/<model>/predictions.csv`. Every model gets the **same test rows**. |
| `ml/fairness_eval.py` | The gate. Fairlearn metrics per protected attribute, CIs, group-size rules, report JSON. Exit 0 = pass, 1 = threshold violated, 2 = bad input/config. |
| `ml/confidence.py` | Wilson score intervals (group rates) and a stratified bootstrap (gaps). |
| `ml/compare.py` | Champion vs. challenger. Trains every configured model, gates each, decides keep/promote, writes `comparison.json`/`.csv`. **Exit code = champion's gate result.** |
| `ml/fairness_thresholds.yaml` | Client-editable thresholds, group-size rules, CI settings, champion/challenger policy, per dataset. |
| `warehouse/schema.sql` | Star schema (idempotent `CREATE IF NOT EXISTS`) plus two Power BI views. |
| `warehouse/etl.py` | Loads one build: dimensions first (MERGE), then that build's facts replaced in one transaction. |
| `.github/workflows/fairness-gate.yml` | CI, including a **Run workflow** button with a champion dropdown. |

### 3.2 How to run it (Windows, PowerShell, from the project folder)

```powershell
.\.venv\Scripts\activate
pip install -r ml\requirements.txt            # first time only
python -m ml.ingest_hmda                       # HMDA row funnel
python -m ml.compare --dataset hmda            # failing build (Vendor A champion) -> exit 1
python -m ml.compare --dataset hmda --champion vendor_a_mitigated   # passing build -> exit 0
python -m ml.train --dataset hmda --model vendor_b_gbm              # one model
python -m ml.fairness_eval --dataset hmda --model vendor_b_gbm      # gate one model
$env:FAIRNESS_SQL_CONNECTION_STRING = '<connection string>'          # session only, never saved
python -m warehouse.etl --dataset hmda --init-schema                 # load latest build
pytest ml/tests warehouse/tests
```

Data files are gitignored. HMDA goes in `data/raw/hmda/hmda_2024_DE.csv`. CI downloads it.

VS Code has local (gitignored) run configurations in `.vscode/launch.json`, e.g.
"HMDA: champion vs challenger (gate)" and "HMDA: champion = mitigated (passing
build)". Use **Ctrl+F5** to skip the debugger. Under F5 the debugger pauses on
`SystemExit: 1`, which is the gate failing *as designed*.

---

## 4. Data methodology

### 4.1 South German Credit (removed)

The first prototype dataset (UCI #573, 1,000 rows). It got the pipeline working end
to end, but with only 300 test rows its confidence intervals were too wide to be
useful, and two datasets made the project harder to explain. **Removed from the
pipeline on 2026-10-16** (§16); the code is in git history before that date.

### 4.2 HMDA, Delaware 2024 (the dataset)

Source: CFPB HMDA Data Browser,
`https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=DE&years=2024`
(48,786 rows × 99 columns, 18.6 MB). Delaware was chosen because it's small but
racially diverse, so race-level fairness checks have enough applicants per group.
Full reasoning: `docs/hmda_data_decisions.md`.

Four questions applied to every row and column:
1. Was a credit decision actually made?
2. Are we comparing like with like?
3. Would the lender know this value at decision time?
4. Is it a protected characteristic, or a stand-in for one?

**Row funnel: 48,786 → 16,511 (84.8% approved)**

| Rows | Step |
|---:|---|
| 48,786 | raw (plan says sample ~50,000 with a fixed seed; DE has fewer, so all kept; the sampler is seeded for bigger states) |
| 33,896 | `action_taken` 1 originated / 2 approved-not-accepted (**approved**), 3 denied (**denied**). Dropped: 4 withdrawn, 5 incomplete, 6 purchased loans, 7/8 preapprovals |
| 23,941 | first liens |
| 22,186 | site-built 1–4 family |
| 18,800 | principal residence |
| 18,546 | not reverse mortgage |
| 17,060 | closed-end (no HELOCs) |
| 17,051 | consumer purpose |
| 16,511 | purchase, refinance, cash-out refinance |

**Features (11):** debt-to-income (bins mapped to midpoints), combined loan-to-value
(1–150 kept), log income, log loan amount, log property value, loan term, loan type,
loan purpose, conforming limit, preapproval requested, has co-applicant. Missing
values (5–6%) are **imputed, not dropped**, because DTI is missing for 12% of denials
vs. 5% of approvals, so dropping rows would bias the sample.

**Dropped, with evidence:**
- **Leakage:** interest rate, rate spread, every loan-cost field (100% missing for
  denials), `denial_reason-*` (code 10 for every approval), `purchaser_type` (non-zero
  only for originated loans).
- **Proxies for race (redlining risk):** census tract, county, MSA, all `tract_*` fields.
- **Lender process:** lender ID (`lei`), AUS system name, credit-score *model type*,
  submission channel.
- **No information after scoping:** constants and fields fixed by the filters.

**Protected audit columns:** `race`, `ethnicity`, `sex` from CFPB's `derived_*`
fields. "Not available" and "Free form text only" become missing and are excluded
from that attribute's audit. `age_group` is **62+ / under 62** from
`applicant_age_above_62`, because Regulation B defines an "elderly" applicant as 62+.
Small groups are **kept, not merged**: merging would hide disparities. The gate
decides what is large enough to gate instead.

**Known limitations:** public HMDA has **no credit score**, so models predict lender
decisions only moderately (AUC 0.81–0.87) and approve about 95% of test applicants
versus 85% actually approved. Applicants who didn't report race have the lowest
approval rate (76%) and can't be audited by race.

---

## 5. Models and the math

Every model trains on the same HMDA training rows and is scored on the same test rows.
Protected attributes are **never inputs** to the honest models; they
are audit columns used afterwards to compare groups *within* each dataset.

Preprocessing (shared): numeric features get a median imputer with a missing-value
indicator, then standardization. Categorical features are converted to text, with
missing as its own category, then one-hot encoded.

| Model | Type | Math / behavior |
|---|---|---|
| `vendor_a_logreg` (Vendor A, **default champion**) | Logistic regression | P(approve) = 1 / (1 + e^−(w·x + b)). Weights fitted by maximum likelihood (minimizing log-loss) with scikit-learn's default L2 penalty. Approve if P ≥ 0.5. |
| `vendor_b_gbm` (Vendor B) | Histogram gradient boosting | 200 shallow trees (learning rate 0.05), each fitted to the gradient of the log-loss left by the previous trees, then summed. Captures non-linear interactions (e.g. high DTI is fine if LTV is low). |
| `vendor_a_mitigated` (mitigated challenger) | Vendor A + Fairlearn `ThresholdOptimizer` | Keeps Vendor A's scores but learns **group-specific (sometimes randomized) thresholds** that maximize accuracy subject to **equalized odds** (equal TPR and FPR across groups). Groups are the **intersection of all protected attributes** (e.g. `White|Not Hispanic|Male|under 62`); intersections with < 100 training rows are pooled. Its "probability" is the chance the randomized threshold approves, so its AUC (0.638) isn't comparable to the others. |

---

## 6. The fairness gate

### 6.1 Metrics (per protected attribute)

- **Demographic parity difference (gated):** max over groups of P(ŷ=1 | group)
  minus the min, i.e. the approval-rate gap.
- **Equalized odds difference (gated):** max of (TPR gap, FPR gap) across groups.
  TPR = share of truly-approved applicants the model approves; FPR = share of
  truly-denied applicants the model wrongly approves.
- **Recorded per group, not gated:** accuracy, precision, recall, false positive
  rate, selection rate, count.

### 6.2 Rules

- **Thresholds:** both gated metrics have `max: 0.10`. This is a placeholder common
  starting point, not a legal standard, and should be set with the client.
- **Group eligibility:** HMDA groups need **≥ 100 rows** (`min_group_size`) to be
  gated, and for equalized odds **≥ 30 truly-approved and ≥ 30 truly-denied**
  (`min_outcome_count`). Smaller
  groups are reported with CIs but can't fail the build. Rationale: an FPR from 1–2
  denied applicants swings between 0 and 1. This was observed; see errors §11.
- **Confidence intervals (95%):**
  - **Group rates:** Wilson score interval: centre = (p + z²/2n)/(1 + z²/n),
    half-width = z·√(p(1−p)/n + z²/4n²)/(1 + z²/n). Accurate for small n and
    rates near 0 or 1. Exact 0/1 at the edges.
  - **Gaps:** **stratified percentile bootstrap**, 1,000 resamples, seed 42.
    Resample rows within each group (sizes fixed), recompute the gap, and take the
    2.5th and 97.5th percentiles.
- **`gate_on`:** `point_estimate` (default: fail if the measured gap exceeds max) or
  `ci_upper` (stricter: fail unless the whole CI is under max).
- **Exit codes:** 0 pass · 1 threshold violated (each failure printed with metric,
  attribute, value, CI, and the two groups behind the gap) · 2 bad config/input.

Example failure line:
```
FAIL equalized_odds_difference on 'race': 0.154 > max 0.100 (95% CI 0.062-0.240) -- false_positive_rate Black or African American=0.787 vs Asian=0.941
```

### 6.3 Report format

`fairness_report.json` holds **one record per metric × group × run**. Fields:
`run_id, created_at, dataset, model, attribute, group, metric, value, ci_lower,
ci_upper, gated, threshold, passed, n, n_excluded_missing, gate_eligible`, plus
`groups_gated`, `groups_skipped` and the widest-gap groups for gated rows. Gated
gaps use `group = "all"`.

---

## 7. Champion vs. challenger

Policy (in `fairness_thresholds.yaml`): champion `vendor_a_logreg`; challengers
`vendor_b_gbm`, `vendor_a_mitigated`; `max_accuracy_drop: 0.05`.

Decision rule (`ml/compare.py::decide`):
1. A challenger **qualifies** only if it passes the gate **and** its accuracy is at
   least champion accuracy − 0.05.
2. If none qualifies, **keep** the champion. If the champion also fails, the reason
   says "escalate for review".
3. If the champion passes and is at least as accurate as the best qualifier, **keep** it.
4. Otherwise **promote** the most accurate qualifier.

The **build result is the champion's gate result**, because the champion is the
model in production. `--champion NAME` (or the CI dropdown) overrides it, and the
configured champion is then compared as a challenger. All models in one build share
**one run ID** (`GITHUB_RUN_ID.GITHUB_RUN_ATTEMPT` in CI, `local-<UTC timestamp>`
locally).

---

## 8. Results

### 8.1 HMDA (4,954 test applications)

| Model | Accuracy | AUC | Approves | Gate |
|---|---|---|---|---|
| vendor_a_logreg (champion) | 0.868 | 0.808 | 95.2% | **FAIL ×3**: EOD race 0.154 (CI 0.062–0.240), ethnicity 0.130, age 0.174 |
| vendor_b_gbm | 0.900 | 0.865 | 93.6% | FAIL ×2: EOD race 0.113 (CI 0.041–0.295), ethnicity 0.107 |
| vendor_a_mitigated | 0.846 | 0.638* | 94.3% | **PASS** all 8 gated metrics |

\*not comparable, see §5. Decision: **PROMOTE `vendor_a_mitigated`** (accuracy 0.846
vs. champion 0.868, floor 0.818).

Groups too small to gate (reported only): race: 2 or more minority races, American
Indian or Alaska Native, Joint, Native Hawaiian or Other Pacific Islander; ethnicity: Joint.

**Mitigation experiments that informed the design:**

| Sensitive feature for ThresholdOptimizer | Objective | Accuracy | Approves | Gate failures |
|---|---|---|---|---|
| race only | balanced accuracy | 0.776 | 77% | 3 (sex, age) |
| race only | accuracy | 0.858 | 98% | 1 (age) |
| race + age | accuracy | 0.858 | 98% | 0 (but near rubber-stamp) |
| race + ethnicity + sex + age (**chosen**) | accuracy | 0.846 | 94% | **0** |

Equalizing on one attribute just moved the disparity to another, so the final
mitigation is intersectional, with an approval rate close to the champion's.

---

## 9. Data warehouse (Azure SQL)

**Server:** `fairness-gates-ab.database.windows.net` · **Database:** `FairnessGates` ·
free serverless offer (auto-pauses when idle; set to auto-pause at the free limit, no
billing) · resource group `fairness-gates-rg`. Created in a region allowed by the
Azure for Students policy; DNS resolves to West US.

### 9.1 Star schema (`warehouse/schema.sql`)

**Grain (declared first, per Kimball):**
- `FactFairnessAudit`: one row per **build × dataset × model × protected group × metric**.
  Measures: `MetricValue, CiLower, CiUpper, Threshold, Passed, GateEligible, GroupSize, RowsMissingAttr`.
- `FactModelEvaluation`: one row per **build × dataset × model**. Measures:
  `Role, TestRows, Accuracy, RocAuc, ApprovalRate, GatePassed, GateFailures, Recommended`.

**Dimensions:** integer surrogate keys; natural keys are UNIQUE.

| Dimension | Natural key | Notes |
|---|---|---|
| `DimDate` | `DateKey` (yyyymmdd) | calendar attributes |
| `DimRun` | `RunId` | the build: started at, commit SHA, branch, trigger, champion, **BuildPassed** |
| `DimDataset` | `DatasetName` | |
| `DimModel` | `ModelName` | vendor, description; **SCD type 1** (overwrite) |
| `DimGroup` | (`AttributeName`, `GroupName`) | `all` = the between-group gap |
| `DimMetric` | `MetricName` | `IsGated`, level (group/gap) |

Champion history isn't a model attribute: it's stored per build (`DimRun.ChampionModel`,
`FactModelEvaluation.Role`), so type 1 loses nothing. Views: `vw_GatedMetrics`,
`vw_BuildSummary`. Unique constraints on each fact's grain enforce "no duplicates".

### 9.2 ETL (`warehouse/etl.py`)

1. **Extract:** read `comparison.json` (with its `run` block) and each model's
   `fairness_report.json`. Reject any report from a different run.
2. **Transform:** build dimension rows and fact rows keyed by natural keys. NaN → NULL.
3. **Load:** MERGE each dimension (insert, or overwrite non-key columns), look up
   surrogate keys, then in one transaction **delete this build's fact rows and
   insert them again**. Rolls back on any error. Re-running is idempotent (verified:
   loading twice leaves 392 rows).
- The connection string comes only from `FAIRNESS_SQL_CONNECTION_STRING` (a GitHub
  Secret in CI, a session variable locally) and is never written to disk.
- **Connection retries:** 4 attempts × 20 s, because the paused free database takes
  about 1 minute to wake.
- **Malformed connection strings fail immediately** with the driver's reason.
- **Passwords are redacted** (`Pwd=`/`Password=`) from every printed error.
- Driver: Microsoft `mssql-python` 1.16.0. It does **not** accept ODBC-only keywords
  such as `Connection Timeout`. Working format (one line, **no surrounding quotes**):
  ```
  Server=tcp:fairness-gates-ab.database.windows.net,1433;Database=FairnessGates;Uid=<admin-login>;Pwd=<password>;Encrypt=yes;TrustServerCertificate=no;
  ```

### 9.3 Security

- **GitHub secret `FAIRNESS_SQL_CONNECTION_STRING`:** uses the **admin** login. CI
  needs to create tables and write rows.
- **Power BI** uses the read-only contained user **`dashboard_reader`**
  (`db_datareader` role), created in the Azure Query editor.
- No credentials are in the repo. `.gitignore` covers `.env*`, keys, and `.azure/`.

### 9.4 Builds currently in Azure

| Build | Champion | BuildPassed |
|---|---|---|
| laptop run (`local-…`) | vendor_a_mitigated | ✅ |
| CI run #12 | vendor_a_mitigated | ✅ |
| CI run #13 (`38924898313.1`) | vendor_a_logreg | ❌ (392 audit rows loaded anyway) |

Runs #10 and #11 aren't in Azure: their load failed on a mis-formatted secret.

---

## 10. CI/CD (GitHub Actions)

Workflow `fairness-gate` runs on push, pull request, and **workflow_dispatch** (the
"Run workflow" button, with a `champion` dropdown). Python 3.12 on ubuntu-latest, pip cache.

| Step | Detail |
|---|---|
| Fetch dataset | HMDA from the CFPB API; a hash change (`305649…2e30`) is only a **warning**, since CFPB can republish. |
| Unit tests | `pytest ml/tests warehouse/tests` (runs with real data, so the real-file tests run too) |
| Ingest HMDA | prints the row funnel |
| Champion vs challenger + fairness gate | `python -m ml.compare --dataset hmda [--champion X]`; exit code = champion's gate |
| Load audit into Azure SQL | runs **even when the gate fails** (`!cancelled()`), only if the secret exists and `comparison.json` was produced; installs `libltdl7 libkrb5-3 libgssapi-krb5-2` for the driver |
| Upload reports | `fairness-reports` artifact, always |

**Demo backups:** run **#12** (champion `vendor_a_mitigated`): all green.
Run **#13** (champion blank): red at the gate, audit still loaded.

---

## 11. Errors encountered and how they were fixed (chronological)

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `.venv/Scripts/python` not recognized in PowerShell | Terminal was in `C:\Users\Chips`; the project lived in a temporary session folder; bash-style path | Moved the project to `OneDrive\Documents\CSC Folder\Projects`, rebuilt `.venv`, use `.\.venv\Scripts\python` |
| 2 | "Per CLAUDE.md" but no CLAUDE.md found | It lived in the OneDrive Projects folder | Found it; aligned code (`approved` 0/1, tests in `ml/tests/`, pinned requirements, `<25/25+`) |
| 3 | 4 gate tests failed | Tests used the repo thresholds (which include `age_group`) on fake data without that column | Tests use a sex-only config |
| 4 | Rewriting git history blocked | Auto-mode permission classifier denied `filter-branch` | Added a separate `chore: expand .gitignore` commit. Note: the `feat: baseline model` message wrongly claims it added the `ml/reports/` ignore |
| 5 | First push rejected | GitHub repo already had an initial commit (README + .gitignore) | Merged unrelated histories, combined both .gitignore files (`data/*` + `!data/README.md`) |
| 6 | VS Code debugger stops on `SystemExit: 1` | Gate exits 1 by design; "Uncaught Exceptions" breakpoint is on | Expected. Press F5 to continue, or Ctrl+F5 to run without debugging |
| 7 | Gate looked frozen in VS Code | Fairlearn import takes about 14 s from OneDrive; slower under the debugger | Expected. Wait, or run without debugging |
| 8 | Pylance warnings (`Hashable` not `str`), twice | pandas labels typed as `Hashable` | Cast with `str(...)`; loop over `RECORDED_METRICS` keys |
| 9 | German training crashed (`fill_value='missing'` vs int) | German categoricals are integer codes | Convert categoricals to text first (`FunctionTransformer(_as_text)`) |
| 10 | Many HMDA gate failures from tiny groups | NHPI ≈ 5 test rows; FPR from 1–2 denials | `min_group_size` / `min_outcome_count`; small groups reported, not gated |
| 11 | Mitigating on race alone failed sex/age | Disparity shifted to other attributes | Intersectional ThresholdOptimizer (all four attributes), accuracy objective |
| 12 | Wilson lower bound 2.8e-17, not 0 | Floating-point rounding | Exact 0/1 bounds at the edges |
| 13 | CI test expected a CI for precision with no predicted approvals | Precision undefined (no denominator); `None` is correct | Test corrected |
| 14 | One comparison would appear as 4 builds | Each model's gate stamped its own timestamp | One shared run ID per build plus a `run` metadata block |
| 15 | Overriding champion dropped Vendor A | Challenger list excluded the configured champion | The configured champion becomes a challenger when overridden |
| 16 | Local `.pbids` file invalid JSON | Shell mangled the `\` in `(localdb)\MSSQLLocalDB` | Rewrote the file directly |
| 17 | Azure: "subscription does not have access … in the selected region", then `RequestDisallowedByAzure` | Azure for Students allowed-regions policy | Created the server in an allowed region |
| 18 | ETL `ConnectionStringParseError` (retried 4×) | `mssql-python` rejects `Connection Timeout=60;` | Removed it from docs; parse errors fail fast; passwords redacted (`752be10`) |
| 19 | "No such host is known" | Wrong server name in the string | Use `fairness-gates-ab.database.windows.net` |
| 20 | "Login failed for user '\<admin-login\>'" | Placeholders pasted literally | Real admin login and password |
| 21 | `CREATE: command not found` | SQL typed into Cloud Shell (bash); the password was also exposed in chat | Ran in the database **Query editor**; password rotated; cleared shell history |
| 22 | `CREATE USER` said the user exists | Deleting the saved query doesn't drop the user | `ALTER USER dashboard_reader WITH PASSWORD = …` |
| 23 | Unclear "Repository access" option | Codespaces/organization secrets page | Use **repo Settings → Secrets and variables → Actions → Repository secrets** |
| 24 | **Every CI run failed at Fetch datasets** (since the first run) | German UCI URL (`…/573/south+german+credit.zip`) returned 404 | `…/573/south+german+credit+update.zip`, verified identical SHA-256 (`dfedde9`) |
| 25 | CI load: `keyword ''' has no value … Unknown keyword ''server'` | GitHub secret pasted with surrounding single quotes | Secret re-saved without quotes. Run #12 green |

---

## 12. Decision log

| Decision | Options considered | Chosen | Why |
|---|---|---|---|
| Unknown sex in German Credit | Map code 2 to female (common) / drop rows / missing | **Missing** | Code 2 mixes women and single men; guessing is wrong |
| Protected attributes as features | Include / exclude | **Exclude** (audit only) | Fairness check; avoid disparate treatment |
| HMDA state/year | Delaware / Maryland / Vermont-Wyoming; 2024 / 2023 | **Delaware 2024** | Small download, diverse enough for race checks, recent stable release |
| HMDA approval definition | Originated only / + approved-not-accepted | **Actions 1 + 2 approved, 3 denied** | The lender said yes in both cases |
| HMDA scope | All products / one product | **First-lien, site-built, owner-occupied, closed-end, consumer, purchase/refi** | Like-with-like underwriting |
| Missing feature values | Drop rows / impute | **Median impute + indicator** | Dropping removes denials disproportionately |
| Age split (HMDA) | HMDA bins / <25 / 62+ | **62+ vs under 62** | Reg B "elderly" definition |
| Small groups | Merge into "other" / drop / gate anyway / report only | **Report with CI, gate only if large enough** | Merging hides disparities; tiny groups are noise |
| Showing pass and fail | Loosen thresholds / mitigated model | **Baseline vs. mitigated, same thresholds** | Honest evidence: the gate catches a problem and approves a fix |
| Mitigation method | Race-only / intersectional; balanced accuracy / accuracy | **Intersectional, accuracy** | Only option passing every gate with an approval rate near the champion's |
| Deliberately biased vendor | Proxy (tract minority %) / direct penalty / none | **Removed** (was a direct logit penalty) | Simplified to two models plus one fix; the gate's failure path is still shown by the champion and tested on fixture data |
| Number of datasets | German + HMDA / HMDA only | **HMDA only** | Professor feedback: one dataset is clearer; German's 300 test rows gave CIs too wide to use |
| Build result | Any model fails / champion fails | **Champion's result** | The champion is what's in production; challengers are advisory |
| CI for gaps | Normal approximation / bootstrap | **Stratified bootstrap** | Max-minus-min has no closed form |
| CI for rates | Normal (Wald) / Wilson | **Wilson** | Correct for small groups and extreme rates |
| HMDA hash in CI | Strict / warn | **Warn** | CFPB may republish; strict would break demos |
| Dashboard tool | Tableau Public + CSV / Power BI / Tableau license | **Power BI Desktop** | Native Azure SQL connection, free, Microsoft stack |
| SQL driver | pyodbc / mssql-python | **mssql-python** | Named in the learning plan; bundles its driver |
| DimModel SCD type | 1 / 2 | **Type 1** | Champion history lives per build, so nothing is lost |
| Fact reload strategy | MERGE per row / delete+insert per build | **Delete+insert in one transaction** | Simple, atomic, idempotent |
| Azure auth for CI | Entra ID / SQL auth | **SQL auth (admin) in a GitHub Secret; read-only user for Power BI** | Simplest for Actions; least privilege for the dashboard |
| VS Code settings | Commit / local only | **Local (gitignored)** | `.vscode/` is in `.gitignore`; can be committed later if wanted |
| `.venv` location | OneDrive / outside | **OneDrive (user's choice)** | Note: slows imports (Fairlearn about 14 s) |

---

## 13. Commit history (all on `main`, pushed)

| Commit | Date | Message |
|---|---|---|
| `41bb5ab` | 10-13 | Initial commit (user, on GitHub) |
| `13bca48` | 10-13 | Update .gitignore to include various patterns (user, on GitHub) |
| `92a4c45` | 10-13 | feat: ingest South German Credit |
| `ca18f09` | 10-13 | feat: baseline model |
| `a32a011` | 10-13 | feat: fairness gate |
| `2d55a22` | 10-13 | test: gate pass/fail cases |
| `b825ff3` | 10-13 | ci: fairness gate workflow |
| `5fe753d` | 10-13 | chore: expand .gitignore |
| `e5dd534` | 10-13 | docs: add project guide |
| `359c10a` | 10-13 | Merge remote main (initial README and .gitignore) |
| `40e73c0` | 10-15 | fix: resolve Pylance type warnings in fairness_eval |
| `59f8b78` | 10-15 | feat: ingest HMDA (Delaware 2024) |
| `d4b48c2` | 10-15 | feat: common model interface and stand-in vendor models |
| `54ce703` | 10-15 | feat: confidence intervals and group-size rules in the gate |
| `9353a02` | 10-15 | feat: champion vs challenger comparison |
| `4258677` | 10-15 | ci: gate the HMDA champion |
| `38e4511` | 10-15 | docs: update project guide for HMDA and new modules |
| `580d41a` | 10-15 | fix: resolve Pylance warning in fairness_eval group loop |
| `3739393` | 10-16 | fix: share one run id per build and record build metadata |
| `e18b764` | 10-16 | feat: star schema and ETL for Azure SQL |
| `0828fab` | 10-16 | ci: load each build's audit into Azure SQL |
| `2d9dd3f` | 10-16 | docs: Azure setup and Power BI dashboard guides |
| `752be10` | 10-16 | fix: fail fast on malformed connection strings and redact passwords |
| `dfedde9` | 10-16 | ci: fix South German Credit download URL |
| `4c0474f` | 10-16 | docs: point Azure Power BI file at fairness-gates-ab |

All commits made with Claude Code carry `Co-Authored-By: Claude Opus 5.5`. Untracked
personal files in the folder (notebooks, learning-plan PDF, CS club text, Morse code
page, a PNG) are deliberately **not** committed. Avoid `git add .`.

---

## 14. Testing

`pytest ml/tests warehouse/tests`: **123 passed, 1 skipped**.

| File | Covers |
|---|---|
| `ml/tests/test_ingest_hmda.py` | action mapping, scope filters, DTI parsing, invalid numbers, protected attributes, seeded sampling, real-file funnel (16,511 rows) |
| `ml/tests/test_models.py` | interface, honest models ignore protected columns, mitigation, reproducibility |
| `ml/tests/test_train.py` | split, determinism, schema, all models share test rows |
| `ml/tests/test_confidence.py` | Wilson textbook values and edges, bootstrap coverage, determinism, undefined rates |
| `ml/tests/test_fairness_eval.py` | config loading and validation, hand-computed metrics, group-size rules, CIs, `gate_on`, report grain |
| `ml/tests/test_gate.py` | runs the gate as a subprocess on biased/fair fixture CSVs: exit 1 naming sex and groups / exit 0 |
| `ml/tests/test_compare.py` | decision rule, policy loading, end-to-end exit code, shared run ID |
| `warehouse/tests/test_etl.py` | extract/transform, NULL handling, run-ID mismatch, redaction, fail-fast parse errors; **opt-in** DB test (idempotency, SCD 1) when `FAIRNESS_TEST_SQL_CONNECTION_STRING` is set (LocalDB `FairnessGatesTest`) |

Local SQL Server **LocalDB** (`(localdb)\MSSQLLocalDB`) has databases `FairnessGates`
(two local builds) and `FairnessGatesTest`. `warehouse/FairnessGates_local.pbids`
opens Power BI on LocalDB.

---

## 15. Open items / next steps

1. **Power BI dashboard (in progress).** Install Power BI Desktop (Microsoft Store),
   double-click `warehouse/FairnessGates_azure.pbids`, sign in as `dashboard_reader`
   (Database tab), load the 8 tables, and confirm `DimRun` has 3 rows. Then
   Phase 2: relationships, mark `DimDate` as the date table, add the measures. Then
   the pages: fairness over time (CI error bars + 0.10 line), pass/fail by build,
   per-vendor comparison. All in `docs/powerbi_dashboard.md`. Save as
   `dashboard/FairnessGates.pbix`.
2. **Demo (Fri Oct 16):** builds #12 (pass) and #13 (fail) are the recorded backups.
3. **Optional hardening:** make the ETL strip stray quotes and whitespace around the
   connection string (error #25), plus a test.
4. **CI maintenance:** GitHub warns that Node.js 20 actions (`checkout@v4`,
   `setup-python@v5`, `upload-artifact@v4`) are deprecated, and `ubuntu-latest`
   moves to Ubuntu 26 from Oct 19, 2026. Bump action versions and re-check the apt
   package names.
5. **Docs nit:** `docs/PROJECT_GUIDE.md` still says "Swapping datasets should only
   touch `ml/ingest.py`". In practice it's the ingest layer (`ingest.py` +
   `ingest_<dataset>.py`).
6. **Later (from the plan):** `pipeline/run_gate.js` (Node orchestration/summary),
   Azure Blob Storage.
7. **Thresholds:** confirm the 0.10 limits and the 5-point accuracy tolerance with the client.

---

## 16. Simplification (2026-10-16, after the professor review)

Feedback: two datasets and four models made the story hard to follow ("in ML you
train, validate and test on one dataset"). The project now presents and runs:

- **One dataset:** HMDA, Delaware 2024.
- **Two models:** logistic regression (Vendor A, champion) and gradient-boosted trees (Vendor B).
- **One fix:** Vendor A mitigated with Fairlearn ThresholdOptimizer, to show what passing looks like.
- **One rule:** if a gated fairness gap is over 0.10 on any attribute, the build fails.
  The 95% CI is reported next to every gap (`gate_on: point_estimate`).

Removed: South German Credit (ingest code, CI download, tests, config section) and
`vendor_c_biased`. HMDA results are unchanged (§8.1). Builds already in Azure still
contain `vendor_c_biased` rows; the dashboard slicer lists it until those builds are
deleted.

Open from the same review: there is no separate validation split (70/30 train/test).
A 60/20/20 train/validation/test split would let the champion/challenger decision
use validation data and keep the test set for the final gate.
