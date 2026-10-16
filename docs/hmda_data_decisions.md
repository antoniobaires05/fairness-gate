# HMDA data decisions

How `ml/ingest_hmda.py` turns the raw HMDA Loan/Application Register (LAR) into the
model-ready frame, and why. Source: CFPB HMDA Data Browser, **Delaware, 2024**
(`data/raw/hmda/hmda_2024_DE.csv`, 48,786 rows x 99 columns, all read as text).

The guiding questions for every row and column:

1. **Was a credit decision actually made?** Fairness of a decision can only be measured where one exists.
2. **Are we comparing like with like?** Different loan products are underwritten under different rules.
3. **Would the lender know this value at decision time?** Anything recorded after the decision leaks the answer.
4. **Is this a protected characteristic, or a stand-in for one?** Those are audit columns, never model inputs.

## Row funnel

Run `python -m ml.ingest_hmda` to reproduce.

| Rows left | Step | Why |
|---:|---|---|
| 48,786 | raw | Learning plan: sample ~50,000 rows with a fixed seed. Delaware has fewer, so every row is kept; the seeded sampler (`SAMPLE_SIZE`, `SEED`) applies to larger states. |
| 33,896 | `action_taken` in 1, 2, 3 | Keep **originated (1)** and **approved but not accepted (2)** as approved, **denied (3)** as denied. Drop **withdrawn (4)** and **incomplete (5)**: no decision was made. Drop **purchased loans (6)**: the reporting lender bought the loan and never made the decision. Drop **preapprovals (7, 8)**: not a loan-application decision. |
| 23,941 | first liens | Second liens are smaller, riskier, priced differently. |
| 22,186 | site-built 1-4 family | Manufactured housing (often chattel loans) and multifamily follow different underwriting. |
| 18,800 | principal residence | Investor and second-home loans have different risk and rules. |
| 18,546 | not reverse mortgage | Reverse mortgages have no repayment underwriting. |
| 17,060 | closed-end | HELOCs (open-end lines) are a different product. |
| 17,051 | consumer purpose | ECOA / Reg B consumer credit; business-purpose loans excluded. |
| **16,511** | purchase, refinance, cash-out refinance | Home improvement and "other" purposes are small and heterogeneous. |

Result: 16,511 applications, **84.8% approved**.

## Target

`approved = 1` for action 1 or 2, `0` for action 3. "Approved but not accepted"
counts as approved because the lender said yes; the applicant walking away is not
a credit decision.

## Columns kept as model features

| Feature | Source | Why | Cleaning |
|---|---|---|---|
| `debt_to_income_ratio` | DTI | Core underwriting factor: can the applicant afford the payment? | Bins (`<20%`, `20%-<30%`, `30%-<36%`, `50%-60%`, `>60%`) mapped to midpoints; exact values 36-49 kept; `Exempt`/blank -> missing |
| `loan_to_value_ratio` | combined LTV | Core factor: how much equity cushions the loan? | Values outside 1-150 treated as data-entry errors -> missing |
| `log_income` | income ($000s) | Repayment capacity | Non-positive -> missing; capped at $5M; log |
| `log_loan_amount` | loan amount | Size of the obligation | log |
| `log_property_value` | property value | Collateral | `Exempt` -> missing; log |
| `loan_term` | months | Payment size and risk horizon | Outside 12-480 -> missing |
| `loan_type` | conventional / FHA / VA / USDA | Different programs, different eligibility | categorical |
| `loan_purpose` | purchase / refi / cash-out | Cash-out refis carry more risk | categorical |
| `conforming_loan_limit` | conforming or not | Jumbo loans are underwritten more strictly | blank -> `unknown` |
| `preapproval` | requested or not | Applicant went through pre-screening first | categorical |
| `has_coapplicant` | from `co-applicant_age` (9999 = none) | A second income on the file | yes / no |

Missing values (5-6% for income, DTI, LTV) are **kept, not dropped**: dropping them
would remove denials disproportionately (DTI is missing for 12% of denials vs 5% of
approvals) and bias the sample. Models impute the median and add a "was missing"
indicator.

## Columns kept as protected audit attributes (never features)

| Attribute | Source | Groups | Not provided -> |
|---|---|---|---|
| `race` | `derived_race` | White, Black or African American, Asian, Joint, American Indian or Alaska Native, Native Hawaiian or Other Pacific Islander, 2 or more minority races | `Race Not Available`, `Free Form Text Only` -> excluded from the race audit |
| `ethnicity` | `derived_ethnicity` | Hispanic or Latino, Not Hispanic or Latino, Joint | `Ethnicity Not Available`, `Free Form Text Only` -> excluded |
| `sex` | `derived_sex` | Male, Female, Joint | `Sex Not Available` -> excluded |
| `age_group` | `applicant_age_above_62` | 62+, under 62 | `NA` -> excluded |

- **Why the derived fields:** CFPB's `derived_*` columns combine applicant and
  co-applicant answers into one documented value. The raw `applicant_race-1..5` fields
  allow up to five answers each and are harder to audit consistently.
- **Why 62:** Regulation B defines an "elderly" applicant as 62 or older, the
  legally meaningful age line for a fair-lending audit.
- **Small groups are kept, not merged.** Merging distinct groups ("other") would hide
  disparities. Instead the gate reports every group with a confidence interval and
  only *gates* on groups with enough applicants (`min_group_size`,
  `min_outcome_count` in `ml/fairness_thresholds.yaml`).
- **Limitation:** applicants who did not report race have the *lowest* approval rate
  (76%). They cannot be audited by race, which may understate disparities.

## Columns dropped, and why

**Leakage: only known after the decision.** These are filled in only for approved
loans (or only for denials), so a model would read the answer off them:

- `interest_rate`, `rate_spread`, `total_loan_costs`, `total_points_and_fees`,
  `origination_charges`, `discount_points`, `lender_credits`: 100% missing for denials.
- `denial_reason-1..4`: code 10 ("not applicable") for every approval.
- `purchaser_type`: non-zero only for originated loans that were later sold.
- `hoepa_status`, `prepayment_penalty_term`, `intro_rate_period`,
  `initially_payable_to_institution`: set at or after closing.

**Protected characteristics and their raw forms:** `applicant_race-*`,
`applicant_ethnicity-*`, `applicant_sex`, `applicant_age`, all co-applicant
equivalents, and the `*_observed` flags (whether the lender recorded them by visual
observation).

**Proxies for protected characteristics (redlining risk):** `census_tract`,
`county_code`, `derived_msa-md`, `tract_minority_population_percent`,
`tract_to_msa_income_percentage`, `tract_population`, `tract_owner_occupied_units`,
`tract_one_to_four_family_homes`, `tract_median_age_of_housing_units`,
`ffiec_msa_md_median_family_income`. Neighborhood characteristics correlate strongly
with race; using them is how redlining shows up in a model.

**Lender process, not applicant risk:** `lei` (which lender: would teach the model
each lender's policy), `aus-1..5` (which underwriting system, not its result),
`applicant_credit_score_type` (which scoring model, not the score),
`submission_of_application`.

**No information after scoping:** `activity_year`, `state_code` (constant);
`derived_dwelling_category`, `occupancy_type`, `lien_status`, `reverse_mortgage`,
`open-end_line_of_credit`, `business_or_commercial_purpose`, `construction_method`
(fixed by the scope filters); `total_units` (99%+ single-unit); manufactured-home and
multifamily fields (out of scope); `derived_loan_product_type` (duplicates
`loan_type` + `lien_status`).

**Rarely populated or not applicable:** `negative_amortization`,
`interest_only_payment`, `balloon_payment`, `other_nonamortizing_features`,
`multifamily_affordable_units`.

## Known limitation

Public HMDA has **no credit score** (the strongest real underwriting factor), so
models predict lender decisions only moderately (ROC AUC 0.81-0.87) and lean toward
approving: about 95% of test applicants versus 85% actually approved. The gate still
measures whether *errors* fall evenly across groups, which is what equalized odds
checks.
