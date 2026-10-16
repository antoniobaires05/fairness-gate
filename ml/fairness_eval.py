"""Fairness gate: audit model predictions per protected attribute.

Usage:
    python -m ml.fairness_eval [--predictions PATH] [--thresholds PATH] [--report PATH]

Reads ml/reports/predictions.csv (from ml/train.py) and ml/fairness_thresholds.yaml,
computes Fairlearn metrics for each protected attribute, and writes
ml/reports/fairness_report.json: one record per metric x group x run.

Exit codes:
    0  every gated metric is within its threshold
    1  at least one gated metric exceeds its threshold (each failure is printed)
    2  bad input: missing file, malformed thresholds, unknown metric or attribute
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score

ML_DIR = Path(__file__).resolve().parent
PREDICTIONS_PATH = ML_DIR / "reports" / "predictions.csv"
THRESHOLDS_PATH = ML_DIR / "fairness_thresholds.yaml"
REPORT_PATH = ML_DIR / "reports" / "fairness_report.json"

GATED_METRICS = {
    "demographic_parity_difference": demographic_parity_difference,
    "equalized_odds_difference": equalized_odds_difference,
}

# Recorded per group, never gated.
RECORDED_METRICS = {
    "accuracy": accuracy_score,
    "precision": lambda y, p: precision_score(y, p, zero_division=0),
    "recall": lambda y, p: recall_score(y, p, zero_division=0),
    "selection_rate": selection_rate,
    "count": count,
}

# Per-group rates that explain each gated metric (used to name the groups in a failure).
EXPLAINING_RATES = {
    "demographic_parity_difference": {"selection_rate": selection_rate},
    "equalized_odds_difference": {
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
    },
}


class GateConfigError(ValueError):
    """Thresholds or predictions are unusable; the gate cannot run."""


def load_thresholds(path: str | Path = THRESHOLDS_PATH) -> dict:
    """Read and validate the thresholds YAML."""
    try:
        config = yaml.safe_load(Path(path).read_text())
    except FileNotFoundError:
        raise GateConfigError(f"thresholds file not found: {path}") from None
    except yaml.YAMLError as e:
        raise GateConfigError(f"{path}: invalid YAML: {e}") from None

    if not isinstance(config, dict):
        raise GateConfigError(f"{path}: expected a mapping at the top level")

    attributes = config.get("attributes")
    if not attributes or not isinstance(attributes, list):
        raise GateConfigError(f"{path}: 'attributes' must be a non-empty list")

    gated = config.get("gated_metrics")
    if not gated or not isinstance(gated, dict):
        raise GateConfigError(f"{path}: 'gated_metrics' must be a non-empty mapping")
    for name, spec in gated.items():
        if name not in GATED_METRICS:
            raise GateConfigError(
                f"{path}: unknown gated metric '{name}' (known: {sorted(GATED_METRICS)})"
            )
        limit = spec.get("max") if isinstance(spec, dict) else None
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not 0 <= limit <= 1:
            raise GateConfigError(f"{path}: {name}.max must be a number between 0 and 1")

    return {
        "attributes": [str(a) for a in attributes],
        "gated_metrics": {name: float(spec["max"]) for name, spec in gated.items()},
    }


def load_predictions(path: str | Path = PREDICTIONS_PATH) -> pd.DataFrame:
    try:
        preds = pd.read_csv(path)
    except FileNotFoundError:
        raise GateConfigError(f"predictions not found: {path} (run python -m ml.train)") from None
    for col in ("y_true", "y_pred"):
        if col not in preds.columns:
            raise GateConfigError(f"{path}: missing column '{col}'")
    return preds


def _widest_gap(y_true, y_pred, groups, metric: str) -> dict:
    """Find the rate and pair of groups behind a gated metric's value."""
    frame = MetricFrame(
        metrics=EXPLAINING_RATES[metric],
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=groups,
    ).by_group
    rate = str((frame.max() - frame.min()).idxmax())
    values = frame[rate]
    return {
        "rate": rate,
        "high_group": str(values.idxmax()),
        "high_value": float(values.max()),
        "low_group": str(values.idxmin()),
        "low_value": float(values.min()),
    }


def evaluate(preds: pd.DataFrame, thresholds: dict, run_id: str) -> tuple[list[dict], list[dict]]:
    """Compute all metrics. Returns (report records, failed gated records)."""
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records, failures = [], []

    for attribute in thresholds["attributes"]:
        if attribute not in preds.columns:
            raise GateConfigError(f"attribute '{attribute}' is not a column in predictions")

        rows = preds[preds[attribute].notna()]
        n_excluded = len(preds) - len(rows)
        if rows[attribute].nunique() < 2:
            raise GateConfigError(f"attribute '{attribute}' has fewer than 2 groups")
        y_true, y_pred = rows["y_true"], rows["y_pred"]
        groups = rows[attribute].astype(str)

        base = {"run_id": run_id, "created_at": created_at, "attribute": attribute,
                "n_excluded_missing": n_excluded}

        by_group = MetricFrame(
            metrics=RECORDED_METRICS, y_true=y_true, y_pred=y_pred, sensitive_features=groups
        ).by_group
        for group, row in by_group.iterrows():
            for metric, value in row.items():
                records.append({**base, "group": str(group), "metric": metric,
                                "value": float(value), "gated": False, "threshold": None,
                                "passed": None, "n": int(row["count"])})

        for metric, limit in thresholds["gated_metrics"].items():
            value = float(GATED_METRICS[metric](y_true, y_pred, sensitive_features=groups))
            record = {**base, "group": "all", "metric": metric, "value": value,
                      "gated": True, "threshold": limit, "passed": value <= limit,
                      "n": len(rows), **_widest_gap(y_true, y_pred, groups, metric)}
            records.append(record)
            if not record["passed"]:
                failures.append(record)

    return records, failures


def format_failure(r: dict) -> str:
    return (
        f"FAIL {r['metric']} on '{r['attribute']}': {r['value']:.3f} > max {r['threshold']:.3f}"
        f" -- {r['rate']} {r['low_group']}={r['low_value']:.3f}"
        f" vs {r['high_group']}={r['high_value']:.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fairness gate: audit model predictions per protected attribute.")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)

    run_id = os.environ.get("GITHUB_RUN_ID") or datetime.now(timezone.utc).strftime("local-%Y%m%dT%H%M%SZ")
    try:
        thresholds = load_thresholds(args.thresholds)
        records, failures = evaluate(load_predictions(args.predictions), thresholds, run_id)
    except GateConfigError as e:
        print(f"ERROR {e}", file=sys.stderr)
        return 2

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(records, indent=2))

    for r in records:
        if r["gated"] and r["passed"]:
            print(f"PASS {r['metric']} on '{r['attribute']}': {r['value']:.3f} <= max {r['threshold']:.3f}")
    for r in failures:
        print(format_failure(r))
    excluded = {r["attribute"]: r["n_excluded_missing"] for r in records if r["n_excluded_missing"]}
    for attribute, n in excluded.items():
        print(f"note: {n} rows with missing '{attribute}' excluded from its metrics")
    print(f"wrote {args.report}")

    if failures:
        print(f"fairness gate FAILED: {len(failures)} metric(s) over threshold", file=sys.stderr)
        return 1
    print("fairness gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
