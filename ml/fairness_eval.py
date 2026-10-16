"""Fairness gate: audit model predictions per protected attribute.

Usage:
    python -m ml.fairness_eval [--dataset NAME] [--model NAME]
                               [--predictions PATH] [--thresholds PATH] [--report PATH]

Reads ml/reports/<dataset>/<model>/predictions.csv (from ml/train.py) and the
<dataset> section of ml/fairness_thresholds.yaml, computes Fairlearn metrics for each
protected attribute with 95% confidence intervals, and writes fairness_report.json
next to the predictions: one record per metric x group x run.

Groups smaller than min_group_size (or, for equalized odds, with fewer than
min_outcome_count approved or denied true outcomes) are reported but not gated:
a rate measured on a handful of applicants is noise, not evidence.

Exit codes:
    0  every gated metric is within its threshold
    1  at least one gated metric exceeds its threshold (each failure is printed)
    2  bad input: missing file, malformed thresholds, unknown metric or attribute
"""

from __future__ import annotations

import argparse
import json
import math
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

from ml.confidence import bootstrap_gaps, wilson_interval

ML_DIR = Path(__file__).resolve().parent
THRESHOLDS_PATH = ML_DIR / "fairness_thresholds.yaml"
REPORTS_DIR = ML_DIR / "reports"
DEFAULT_DATASET = "german_credit"
DEFAULT_MODEL = "vendor_a_logreg"

GATED_METRICS = {
    "demographic_parity_difference": demographic_parity_difference,
    "equalized_odds_difference": equalized_odds_difference,
}

# Recorded per group, never gated.
RECORDED_METRICS = {
    "accuracy": accuracy_score,
    "precision": lambda y, p: precision_score(y, p, zero_division=0),
    "recall": lambda y, p: recall_score(y, p, zero_division=0),
    "false_positive_rate": false_positive_rate,
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

# Settings a config may omit. Small defaults keep tiny hand-built test files gateable.
DEFAULTS = {
    "min_group_size": 1,
    "min_outcome_count": 1,
    "confidence_level": 0.95,
    "bootstrap_samples": 1000,
    "gate_on": "point_estimate",
}
GATE_ON = ("point_estimate", "ci_upper")


class GateConfigError(ValueError):
    """Thresholds or predictions are unusable; the gate cannot run."""


def predictions_path(dataset: str, model: str) -> Path:
    return REPORTS_DIR / dataset / model / "predictions.csv"


def load_thresholds(path: str | Path = THRESHOLDS_PATH, dataset: str = DEFAULT_DATASET) -> dict:
    """Read the thresholds YAML and resolve the settings for one dataset.

    Two layouts are accepted:
      - flat: attributes, gated_metrics and settings at the top level (one dataset);
      - sectioned: a `defaults` mapping plus `datasets: {<name>: {...}}` overrides.
    """
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except FileNotFoundError:
        raise GateConfigError(f"thresholds file not found: {path}") from None
    except yaml.YAMLError as e:
        raise GateConfigError(f"{path}: invalid YAML: {e}") from None
    if not isinstance(raw, dict):
        raise GateConfigError(f"{path}: expected a mapping at the top level")

    if "datasets" in raw:
        sections = raw.get("datasets") or {}
        if dataset not in sections:
            raise GateConfigError(f"{path}: no section for dataset '{dataset}' under 'datasets'")
        config = {**(raw.get("defaults") or {}), **(sections[dataset] or {})}
    else:
        config = raw
    return _validate(config, path)


def _validate(config: dict, path) -> dict:
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

    settings = {**DEFAULTS, **{k: config[k] for k in DEFAULTS if k in config}}
    for key in ("min_group_size", "min_outcome_count", "bootstrap_samples"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise GateConfigError(f"{path}: {key} must be a positive integer")
    if not 0 < settings["confidence_level"] < 1:
        raise GateConfigError(f"{path}: confidence_level must be between 0 and 1")
    if settings["gate_on"] not in GATE_ON:
        raise GateConfigError(f"{path}: gate_on must be one of {GATE_ON}")

    return {
        "attributes": [str(a) for a in attributes],
        "gated_metrics": {name: float(spec["max"]) for name, spec in gated.items()},
        **settings,
    }


def load_predictions(path: str | Path) -> pd.DataFrame:
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


def _rate_interval(metric: str, y_true, y_pred, level: float) -> tuple[float, float]:
    """Wilson interval for a per-group rate; (nan, nan) for counts."""
    y, p = pd.Series(y_true).astype(bool), pd.Series(y_pred).astype(bool)
    k_n = {
        "accuracy": ((y == p).sum(), len(y)),
        "precision": ((y & p).sum(), p.sum()),
        "recall": ((y & p).sum(), y.sum()),
        "false_positive_rate": ((~y & p).sum(), (~y).sum()),
        "selection_rate": (p.sum(), len(p)),
    }
    if metric not in k_n:
        return math.nan, math.nan
    k, n = k_n[metric]
    return wilson_interval(int(k), int(n), level)


def _eligible_groups(rows: pd.DataFrame, groups: pd.Series, metric: str, cfg: dict) -> list[str]:
    stats = rows.assign(_g=groups).groupby("_g")["y_true"].agg(["size", "sum"])
    ok = stats["size"] >= cfg["min_group_size"]
    if metric == "equalized_odds_difference":
        ok &= (stats["sum"] >= cfg["min_outcome_count"])
        ok &= (stats["size"] - stats["sum"] >= cfg["min_outcome_count"])
    return [str(g) for g in stats.index[ok]]


def _nan_to_none(value):
    return None if isinstance(value, float) and math.isnan(value) else value


def evaluate(preds: pd.DataFrame, thresholds: dict, run_id: str, dataset: str = DEFAULT_DATASET,
             model: str = DEFAULT_MODEL) -> tuple[list[dict], list[dict]]:
    """Compute all metrics. Returns (report records, failed gated records)."""
    cfg = {**DEFAULTS, **thresholds}
    level = cfg["confidence_level"]
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records, failures = [], []

    for attribute in cfg["attributes"]:
        if attribute not in preds.columns:
            raise GateConfigError(f"attribute '{attribute}' is not a column in predictions")

        rows = preds[preds[attribute].notna()]
        n_excluded = len(preds) - len(rows)
        if rows[attribute].nunique() < 2:
            raise GateConfigError(f"attribute '{attribute}' has fewer than 2 groups")
        groups = rows[attribute].astype(str)

        base = {"run_id": run_id, "created_at": created_at, "dataset": dataset, "model": model,
                "attribute": attribute, "n_excluded_missing": n_excluded}

        by_group = MetricFrame(
            metrics=RECORDED_METRICS, y_true=rows["y_true"], y_pred=rows["y_pred"],
            sensitive_features=groups,
        ).by_group
        dpd_eligible = set(_eligible_groups(rows, groups, "demographic_parity_difference", cfg))
        for group, row in by_group.iterrows():
            in_group = groups == group
            for metric, value in row.items():
                lo, hi = _rate_interval(metric, rows.loc[in_group, "y_true"],
                                        rows.loc[in_group, "y_pred"], level)
                records.append({**base, "group": str(group), "metric": metric,
                                "value": float(value), "ci_lower": _nan_to_none(lo),
                                "ci_upper": _nan_to_none(hi), "gated": False, "threshold": None,
                                "passed": None, "n": int(row["count"]),
                                "gate_eligible": str(group) in dpd_eligible})

        for metric, limit in cfg["gated_metrics"].items():
            eligible = _eligible_groups(rows, groups, metric, cfg)
            record = {**base, "group": "all", "metric": metric, "gated": True,
                      "threshold": limit, "groups_gated": eligible,
                      "groups_skipped": sorted(set(groups) - set(eligible))}
            if len(eligible) < 2:
                records.append({**record, "value": None, "ci_lower": None, "ci_upper": None,
                                "passed": None, "n": 0})
                continue
            sub = groups.isin(eligible)
            y_true, y_pred, g = rows.loc[sub, "y_true"], rows.loc[sub, "y_pred"], groups[sub]
            value = float(GATED_METRICS[metric](y_true, y_pred, sensitive_features=g))
            lo, hi = bootstrap_gaps(y_true, y_pred, g, n_boot=cfg["bootstrap_samples"],
                                    level=level)[metric]
            tested = value if cfg["gate_on"] == "point_estimate" else hi
            record.update({"value": value, "ci_lower": _nan_to_none(lo), "ci_upper": _nan_to_none(hi),
                           "passed": bool(tested <= limit), "n": int(sub.sum()),
                           **_widest_gap(y_true, y_pred, g, metric)})
            records.append(record)
            if not record["passed"]:
                failures.append(record)

    return records, failures


def _ci(r: dict) -> str:
    if r.get("ci_lower") is None:
        return ""
    return f" (95% CI {r['ci_lower']:.3f}-{r['ci_upper']:.3f})"


def format_failure(r: dict) -> str:
    return (
        f"FAIL {r['metric']} on '{r['attribute']}': {r['value']:.3f} > max {r['threshold']:.3f}"
        f"{_ci(r)} -- {r['rate']} {r['low_group']}={r['low_value']:.3f}"
        f" vs {r['high_group']}={r['high_value']:.3f}"
    )


def print_summary(records: list[dict], failures: list[dict]) -> None:
    for r in records:
        if not r["gated"]:
            continue
        if r["passed"] is None:
            print(f"SKIP {r['metric']} on '{r['attribute']}': fewer than 2 groups large enough to gate")
        elif r["passed"]:
            print(f"PASS {r['metric']} on '{r['attribute']}': {r['value']:.3f} <= max "
                  f"{r['threshold']:.3f}{_ci(r)}")
    for r in failures:
        print(format_failure(r))
    skipped = {}
    for r in records:
        if r["gated"] and r["groups_skipped"]:
            skipped.setdefault(r["attribute"], set()).update(r["groups_skipped"])
    for attribute, groups in skipped.items():
        print(f"note: '{attribute}' groups too small to gate (reported only): {', '.join(sorted(groups))}")
    excluded = {r["attribute"]: r["n_excluded_missing"] for r in records if r["n_excluded_missing"]}
    for attribute, n in excluded.items():
        print(f"note: {n} rows with missing '{attribute}' excluded from its metrics")


def run_gate(preds: pd.DataFrame, thresholds: dict, report: Path, dataset: str,
             model: str) -> tuple[list[dict], list[dict]]:
    run_id = os.environ.get("GITHUB_RUN_ID") or datetime.now(timezone.utc).strftime("local-%Y%m%dT%H%M%SZ")
    records, failures = evaluate(preds, thresholds, run_id, dataset, model)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(records, indent=2))
    return records, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fairness gate: audit model predictions per protected attribute.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    predictions = args.predictions or predictions_path(args.dataset, args.model)
    report = args.report or predictions.parent / "fairness_report.json"

    try:
        thresholds = load_thresholds(args.thresholds, args.dataset)
        records, failures = run_gate(load_predictions(predictions), thresholds, report,
                                     args.dataset, args.model)
    except GateConfigError as e:
        print(f"ERROR {e}", file=sys.stderr)
        return 2

    print_summary(records, failures)
    print(f"wrote {report}")
    if failures:
        print(f"fairness gate FAILED: {len(failures)} metric(s) over threshold", file=sys.stderr)
        return 1
    print("fairness gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
