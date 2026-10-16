"""Load and clean HMDA Loan/Application Register data (one state, one year).

Source: CFPB HMDA Data Browser, e.g.
    https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=DE&years=2024
saved as data/raw/hmda/hmda_2024_DE.csv. Every column is read as text first so
codes like "1111" (exempt) and "8888" (not applicable) are never mangled.

Why each row and column is kept or dropped is documented in
docs/hmda_data_decisions.md; the short version is in the comments below.

Output (ml.ingest.Dataset):
    approved   -- 1 if action_taken is 1 (originated) or 2 (approved, not accepted),
                  0 if 3 (denied). Every other action is dropped.
    race, ethnicity, sex, age_group -- protected audit columns, <NA> when the
                  applicant did not provide the information. Never model features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.ingest import Dataset

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "hmda" / "hmda_2024_DE.csv"

SEED = 42
SAMPLE_SIZE = 50_000  # learning-plan target; Delaware 2024 has fewer rows, so all are kept

# --- Rows ---------------------------------------------------------------------

APPROVED_ACTIONS = {"1", "2"}  # originated; approved but not accepted by applicant
DENIED_ACTIONS = {"3"}
# Dropped: 4 withdrawn, 5 incomplete (no decision was made), 6 purchased loan (the
# reporting lender never made the decision), 7/8 preapproval requests (not a loan
# application decision).

# Scope: compare like with like -- one standard consumer mortgage product.
# (column, allowed values, reason)
SCOPE_FILTERS = [
    ("lien_status", {"1"}, "first liens only; junior liens are priced/underwritten differently"),
    ("derived_dwelling_category", {"Single Family (1-4 Units):Site-Built"},
     "site-built 1-4 family; manufactured and multifamily follow other rules"),
    ("occupancy_type", {"1"}, "principal residence; investor and second homes differ"),
    ("reverse_mortgage", {"2"}, "reverse mortgages have no repayment underwriting"),
    ("open-end_line_of_credit", {"2"}, "closed-end loans; HELOCs are a different product"),
    ("business_or_commercial_purpose", {"2"}, "consumer loans only (ECOA/Reg B scope)"),
    ("loan_purpose", {"1", "31", "32"}, "home purchase, refinance, cash-out refinance"),
]

# --- Columns ------------------------------------------------------------------

NUMERIC_FEATURES = [
    "log_loan_amount",
    "log_income",
    "log_property_value",
    "loan_to_value_ratio",
    "debt_to_income_ratio",
    "loan_term",
]
CATEGORICAL_FEATURES = [
    "loan_type",           # conventional / FHA / VA / USDA
    "loan_purpose",        # purchase / refi / cash-out
    "conforming_loan_limit",
    "preapproval",         # applicant requested a preapproval first
    "has_coapplicant",     # second income on the application
]
PROTECTED = ["race", "ethnicity", "sex", "age_group"]

# Debt-to-income is reported in bins below 36% and above 49%, exact from 36-49.
DTI_BINS = {"<20%": 15.0, "20%-<30%": 25.0, "30%-<36%": 33.0, "50%-60%": 55.0, ">60%": 65.0}

# derived_* values that mean "not provided" -> <NA>, excluded from that attribute's audit.
NOT_PROVIDED = {"Race Not Available", "Ethnicity Not Available", "Sex Not Available",
                "Free Form Text Only"}


def load_raw(path: str | Path = DEFAULT_PATH, sample_size: int = SAMPLE_SIZE,
             seed: int = SEED) -> pd.DataFrame:
    """Read the LAR CSV as text; sample down to sample_size rows with a fixed seed."""
    df = pd.read_csv(path, dtype=str, low_memory=False)
    required = {"action_taken", "derived_race", "derived_ethnicity", "derived_sex",
                "applicant_age_above_62", *(c for c, _, _ in SCOPE_FILTERS)}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns {sorted(missing)}")
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)
    return df.sort_index()


def filter_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Keep decided applications in scope. Returns the rows and a (step, rows left) log."""
    log = [("raw", len(df))]
    df = df[df["action_taken"].isin(APPROVED_ACTIONS | DENIED_ACTIONS)]
    log.append(("decision made (approved or denied)", len(df)))
    for column, allowed, reason in SCOPE_FILTERS:
        df = df[df[column].isin(allowed)]
        log.append((f"{column}: {reason}", len(df)))
    return df, log


def _numeric(s: pd.Series) -> pd.Series:
    """'Exempt', 'NA' and blanks -> NaN."""
    return pd.to_numeric(s, errors="coerce")


def _dti(s: pd.Series) -> pd.Series:
    mapped = s.map(DTI_BINS)
    return mapped.fillna(_numeric(s))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw text columns into model features. Missing values stay NaN for the model to impute."""
    out = pd.DataFrame(index=df.index)
    income = _numeric(df["income"])  # thousands of dollars; negatives are reporting errors
    out["log_income"] = np.log1p(income.where(income > 0).clip(upper=5_000))
    out["log_loan_amount"] = np.log1p(_numeric(df["loan_amount"]))
    out["log_property_value"] = np.log1p(_numeric(df["property_value"]))
    ltv = _numeric(df["loan_to_value_ratio"])
    out["loan_to_value_ratio"] = ltv.where(ltv.between(1, 150))  # outside = data-entry errors
    out["debt_to_income_ratio"] = _dti(df["debt_to_income_ratio"])
    term = _numeric(df["loan_term"])
    out["loan_term"] = term.where(term.between(12, 480))
    for col in ("loan_type", "loan_purpose", "preapproval"):
        out[col] = df[col]
    out["conforming_loan_limit"] = df["conforming_loan_limit"].fillna("unknown")
    out["has_coapplicant"] = (df["co-applicant_age"] != "9999").map({True: "yes", False: "no"})
    return out


def build_protected(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in ("race", "ethnicity", "sex"):
        out[col] = df[f"derived_{col}"].where(~df[f"derived_{col}"].isin(NOT_PROVIDED)).astype("string")
    # Reg B defines an "elderly" applicant as 62 or older.
    out["age_group"] = df["applicant_age_above_62"].map({"Yes": "62+", "No": "under 62"}).astype("string")
    return out


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    rows, log = filter_rows(raw)
    frame = pd.concat([build_features(rows), build_protected(rows)], axis=1)
    frame["approved"] = rows["action_taken"].isin(APPROVED_ACTIONS).astype(int)
    return frame.reset_index(drop=True), log


def load_dataset(path: str | Path = DEFAULT_PATH) -> Dataset:
    frame, _ = clean(load_raw(path))
    return Dataset(
        name="hmda",
        frame=frame,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        protected=PROTECTED,
    )


if __name__ == "__main__":
    frame, log = clean(load_raw())
    print("row funnel:")
    for step, n in log:
        print(f"  {n:6d}  {step}")
    print(f"\n{len(frame)} rows, approval rate {frame['approved'].mean():.3f}")
    print("\nmissing feature values (imputed at training time):")
    print(frame[NUMERIC_FEATURES].isna().mean().round(3).to_string())
    for col in PROTECTED:
        rates = frame.groupby(col, dropna=False)["approved"].agg(["size", "mean"])
        print(f"\n{col}:\n{rates.round(3).to_string()}")
