"""Train one model on one dataset and save its test-set predictions.

Usage:
    python -m ml.train [--dataset german_credit|hmda] [--model NAME] [--out PATH]

Writes ml/reports/<dataset>/<model>/predictions.csv with one row per test example:
    row_id, y_true, y_pred, y_prob, <protected columns...>

The split is stratified on the target with a fixed seed, so every model on the
same dataset is scored on exactly the same test rows (needed for champion vs.
challenger). Protected columns are never model features (see ml/models.py); they
are carried through so the predictions can be audited.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ml import ingest
from ml.ingest import Dataset
from ml.models import MODELS, CreditModel, build_model

SEED = 42
TEST_SIZE = 0.3
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
DEFAULT_DATASET = "german_credit"
DEFAULT_MODEL = "vendor_a_logreg"


def predictions_path(dataset: str, model: str) -> Path:
    return REPORTS_DIR / dataset / model / "predictions.csv"


def split(dataset: Dataset, seed: int = SEED, test_size: float = TEST_SIZE):
    """Stratified train/test split on the target."""
    frame = dataset.frame
    return train_test_split(
        frame, test_size=test_size, random_state=seed, stratify=frame[dataset.target]
    )


def train(dataset: Dataset, model_name: str = DEFAULT_MODEL,
          seed: int = SEED) -> tuple[CreditModel, pd.DataFrame]:
    """Fit on the train split; return the model and test-set predictions."""
    train_df, test_df = split(dataset, seed=seed)
    model = build_model(model_name, dataset, seed).fit(train_df)

    predictions = pd.DataFrame({
        "row_id": test_df.index,
        "y_true": test_df[dataset.target].astype(int).to_numpy(),
        "y_pred": model.predict(test_df),
        "y_prob": model.predict_proba(test_df),
    })
    for col in dataset.protected:
        predictions[col] = test_df[col].to_numpy()
    return model, predictions.sort_values("row_id", ignore_index=True)


def summarize(predictions: pd.DataFrame) -> dict:
    return {
        "n": len(predictions),
        "accuracy": float(accuracy_score(predictions["y_true"], predictions["y_pred"])),
        "roc_auc": float(roc_auc_score(predictions["y_true"], predictions["y_prob"])),
        "approval_rate": float(predictions["y_pred"].mean()),
    }


def run(dataset: Dataset, model_name: str, out: Path | None = None) -> pd.DataFrame:
    _, predictions = train(dataset, model_name)
    out = out or predictions_path(dataset.name, model_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)
    s = summarize(predictions)
    print(f"{dataset.name}/{model_name}: test n={s['n']}  accuracy={s['accuracy']:.3f}"
          f"  roc_auc={s['roc_auc']:.3f}  approval_rate={s['approval_rate']:.3f}")
    print(f"wrote {out}")
    return predictions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a model and save test-set predictions.")
    parser.add_argument("--dataset", choices=ingest.DATASET_NAMES, default=DEFAULT_DATASET)
    parser.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    run(ingest.load_dataset(args.dataset), args.model, args.out)


if __name__ == "__main__":
    main()
