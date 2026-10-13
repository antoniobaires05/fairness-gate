"""Load and clean the South German Credit dataset (Grömping, 2019).

Source: UCI ML Repository, "South German Credit" -- SouthGermanCredit.asc,
a whitespace-separated file with German column names and integer codes.
See data/raw/codetable.txt for the code meanings.

Derived columns:
    approved   -- 0/1, 1 when credit_risk == 1 ("good"). Note this is the
                  repayment outcome of loans that were granted, not a lending
                  decision; bad risks are oversampled (300 bad / 700 good).
    sex        -- "male" / "female", or <NA> when not identifiable. Code 2 of
                  personal_status_sex means "female : non-single OR male :
                  single", so sex is unknown for those 310 rows. Do not guess.
    age_group  -- ordered categorical built from age with AGE_BINS / AGE_LABELS.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "SouthGermanCredit.asc"

# German column names from the file -> English names (from the dataset's R script).
COLUMN_NAMES = {
    "laufkont": "status",
    "laufzeit": "duration",
    "moral": "credit_history",
    "verw": "purpose",
    "hoehe": "amount",
    "sparkont": "savings",
    "beszeit": "employment_duration",
    "rate": "installment_rate",
    "famges": "personal_status_sex",
    "buerge": "other_debtors",
    "wohnzeit": "present_residence",
    "verm": "property",
    "alter": "age",
    "weitkred": "other_installment_plans",
    "wohn": "housing",
    "bishkred": "number_credits",
    "beruf": "job",
    "pers": "people_liable",
    "telef": "telephone",
    "gastarb": "foreign_worker",
    "kredit": "credit_risk",
}

# Allowed integer codes per categorical column (from codetable.txt).
VALID_CODES = {
    "status": range(1, 5),
    "credit_history": range(0, 5),
    "purpose": range(0, 11),
    "savings": range(1, 6),
    "employment_duration": range(1, 6),
    "installment_rate": range(1, 5),
    "personal_status_sex": range(1, 5),
    "other_debtors": range(1, 4),
    "present_residence": range(1, 5),
    "property": range(1, 5),
    "other_installment_plans": range(1, 4),
    "housing": range(1, 4),
    "number_credits": range(1, 5),
    "job": range(1, 5),
    "people_liable": range(1, 3),
    "telephone": range(1, 3),
    "foreign_worker": range(1, 3),
    "credit_risk": range(0, 2),
}

# personal_status_sex code -> sex. Code 2 is deliberately absent (ambiguous).
SEX_BY_STATUS_CODE = {1: "male", 3: "male", 4: "female"}

AGE_BINS = [0, 25, 200]
AGE_LABELS = ["<25", "25+"]


def load_raw(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Read the .asc file as-is and rename columns to English."""
    df = pd.read_csv(path, sep=r"\s+")
    missing = set(COLUMN_NAMES) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected columns {sorted(missing)}")
    return df[list(COLUMN_NAMES)].rename(columns=COLUMN_NAMES)


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError if the raw data has nulls or out-of-range values."""
    if df.isna().any().any():
        cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"null values in columns {cols}")
    for col, codes in VALID_CODES.items():
        bad = ~df[col].isin(codes)
        if bad.any():
            raise ValueError(f"{col}: unexpected codes {sorted(df.loc[bad, col].unique())}")
    for col in ("duration", "amount", "age"):
        if (df[col] <= 0).any():
            raise ValueError(f"{col}: non-positive values")


def add_derived_columns(
    df: pd.DataFrame,
    age_bins: list[int] = AGE_BINS,
    age_labels: list[str] = AGE_LABELS,
) -> pd.DataFrame:
    """Return a copy of df with approved, sex and age_group added."""
    out = df.copy()
    out["approved"] = out["credit_risk"].eq(1).astype(int)
    out["sex"] = out["personal_status_sex"].map(SEX_BY_STATUS_CODE).astype("string")
    out["age_group"] = pd.cut(
        out["age"], bins=age_bins, labels=age_labels, right=False, ordered=True
    )
    return out


def load(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load, validate and clean the dataset. Main entry point."""
    df = load_raw(path)
    validate(df)
    return add_derived_columns(df)


if __name__ == "__main__":
    data = load()
    print(data.shape)
    print(data[["approved", "sex", "age_group"]].value_counts(dropna=False).sort_index())
