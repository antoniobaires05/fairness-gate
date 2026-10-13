"""Train a logistic regression on South German Credit and save test-set predictions.

Usage:
    python -m ml.train

Writes ml/reports/predictions.csv with one row per test example:
    row_id, y_true, y_pred, y_prob, sex, age_group

Protected attributes (sex via personal_status_sex, and age) are left out of the
features but carried through to the output so the predictions can be audited.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml import ingest

SEED = 42
TEST_SIZE = 0.3
TARGET = "approved"
REPORT_PATH = Path(__file__).resolve().parent / "reports" / "predictions.csv"

NUMERIC_FEATURES = ["duration", "amount"]
# Coded features whose codes are unordered categories -> one-hot encoded.
CATEGORICAL_FEATURES = [
    "status",
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "installment_rate",
    "other_debtors",
    "present_residence",
    "property",
    "other_installment_plans",
    "housing",
    "number_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
]
# Not used as features; passed through to the predictions file.
PROTECTED = ["sex", "age_group"]


def build_model() -> Pipeline:
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )


def split(df: pd.DataFrame, seed: int = SEED, test_size: float = TEST_SIZE):
    """Stratified train/test split on the target."""
    return train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[TARGET]
    )


def train(df: pd.DataFrame, seed: int = SEED) -> tuple[Pipeline, pd.DataFrame]:
    """Fit on the train split; return the model and test-set predictions."""
    train_df, test_df = split(df, seed=seed)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    model = build_model()
    model.fit(train_df[features], train_df[TARGET])

    prob = model.predict_proba(test_df[features])[:, 1]
    predictions = pd.DataFrame(
        {
            "row_id": test_df.index,
            "y_true": test_df[TARGET].astype(int).to_numpy(),
            "y_pred": (prob >= 0.5).astype(int),
            "y_prob": prob,
            "sex": test_df["sex"].to_numpy(),
            "age_group": test_df["age_group"].to_numpy(),
        }
    ).sort_values("row_id", ignore_index=True)
    return model, predictions


def main(path: Path = REPORT_PATH) -> pd.DataFrame:
    _, predictions = train(ingest.load())
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)

    acc = accuracy_score(predictions["y_true"], predictions["y_pred"])
    auc = roc_auc_score(predictions["y_true"], predictions["y_prob"])
    print(f"test n={len(predictions)}  accuracy={acc:.3f}  roc_auc={auc:.3f}")
    print(f"wrote {path}")
    return predictions


if __name__ == "__main__":
    main()
