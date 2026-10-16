"""Champion vs. challenger: train every configured model on the same split, run the
fairness gate on each, and recommend whether to keep or replace the champion.

Usage:
    python -m ml.compare [--dataset hmda] [--champion NAME] [--thresholds PATH]

Champion, challengers and max_accuracy_drop come from ml/fairness_thresholds.yaml.
A challenger is recommended only if it passes the gate AND its accuracy is within
max_accuracy_drop of the champion's; among those, the most accurate wins.

Writes ml/reports/<dataset>/comparison.json and comparison.csv (one row per model),
plus each model's predictions.csv and fairness_report.json.

Exit code is the CHAMPION's gate result (0 pass, 1 fail, 2 config error): the build
reflects the model actually in production. Challenger results are advisory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from ml import ingest, train
from ml.fairness_eval import (
    THRESHOLDS_PATH,
    GateConfigError,
    format_failure,
    load_thresholds,
    run_gate,
)
from ml.models import MODELS

DEFAULT_DATASET = "hmda"


def load_policy(path: Path, dataset: str) -> dict:
    """Champion/challenger settings: defaults merged with the dataset's section."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    merged = {**(raw.get("defaults") or {}), **((raw.get("datasets") or {}).get(dataset) or {})}
    policy = {
        "champion": merged.get("champion"),
        "challengers": list(merged.get("challengers") or []),
        "max_accuracy_drop": float(merged.get("max_accuracy_drop", 0.0)),
    }
    for name in [policy["champion"], *policy["challengers"]]:
        if name not in MODELS:
            raise GateConfigError(f"{path}: unknown model '{name}' (known: {', '.join(MODELS)})")
    return policy


def decide(rows: list[dict], champion: str, max_accuracy_drop: float) -> dict:
    """Pick the recommendation from per-model summary rows."""
    by_name = {r["model"]: r for r in rows}
    champ = by_name[champion]
    floor = champ["accuracy"] - max_accuracy_drop
    eligible = [r for r in rows if r["role"] == "challenger" and r["gate_passed"]
                and r["accuracy"] >= floor]
    best = max(eligible, key=lambda r: r["accuracy"], default=None)

    if best is None:
        reason = ("champion passes the gate; no challenger qualifies" if champ["gate_passed"]
                  else "champion FAILS the gate, but no challenger both passes and stays within "
                       f"{max_accuracy_drop:.0%} accuracy -- escalate for review")
        return {"action": "keep", "model": champion, "reason": reason}
    if champ["gate_passed"] and champ["accuracy"] >= best["accuracy"]:
        return {"action": "keep", "model": champion,
                "reason": "champion passes the gate and is at least as accurate as any passing challenger"}
    why = "champion fails the gate" if not champ["gate_passed"] else "more accurate and also passes"
    return {"action": "promote", "model": best["model"],
            "reason": f"{why}; {best['model']} passes with accuracy {best['accuracy']:.3f} "
                      f"(champion {champ['accuracy']:.3f}, floor {floor:.3f})"}


def compare(dataset_name: str, thresholds_path: Path = THRESHOLDS_PATH,
            champion: str | None = None) -> tuple[list[dict], dict]:
    thresholds = load_thresholds(thresholds_path, dataset_name)
    policy = load_policy(thresholds_path, dataset_name)
    champion = champion or policy["champion"]
    challengers = [m for m in policy["challengers"] if m != champion]
    if champion not in MODELS:
        raise GateConfigError(f"unknown champion '{champion}'")

    dataset = ingest.load_dataset(dataset_name)
    rows = []
    for role, name in [("champion", champion), *[("challenger", m) for m in challengers]]:
        predictions = train.run(dataset, name)
        report = train.predictions_path(dataset_name, name).parent / "fairness_report.json"
        records, failures = run_gate(predictions, thresholds, report, dataset_name, name)
        gated = {r["metric"] + "@" + r["attribute"]: r for r in records if r["gated"]}
        rows.append({
            "run_id": records[0]["run_id"],
            "dataset": dataset_name,
            "model": name,
            "role": role,
            "description": MODELS[name].description,
            **train.summarize(predictions),
            "gate_passed": not failures,
            "n_failures": len(failures),
            "failures": [format_failure(f) for f in failures],
            "gated_metrics": {k: {"value": r["value"], "ci_lower": r["ci_lower"],
                                  "ci_upper": r["ci_upper"], "passed": r["passed"]}
                              for k, r in gated.items()},
        })
    return rows, decide(rows, champion, policy["max_accuracy_drop"])


def write_outputs(rows: list[dict], decision: dict, dataset_name: str) -> Path:
    out_dir = train.REPORTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r["recommended"] = r["model"] == decision["model"]
    (out_dir / "comparison.json").write_text(json.dumps({"decision": decision, "models": rows}, indent=2))
    flat = []
    for r in rows:
        row = {k: v for k, v in r.items() if k not in ("failures", "gated_metrics")}
        for key, m in r["gated_metrics"].items():
            row[key] = m["value"]
        flat.append(row)
    pd.DataFrame(flat).to_csv(out_dir / "comparison.csv", index=False)
    return out_dir


def print_table(rows: list[dict], decision: dict) -> None:
    print(f"\n{'model':22s} {'role':10s} {'accuracy':>8s} {'auc':>6s} {'approve':>7s}  gate")
    for r in rows:
        gate = "PASS" if r["gate_passed"] else f"FAIL ({r['n_failures']})"
        print(f"{r['model']:22s} {r['role']:10s} {r['accuracy']:8.3f} {r['roc_auc']:6.3f} "
              f"{r['approval_rate']:7.3f}  {gate}")
    for r in rows:
        for line in r["failures"]:
            print(f"  [{r['model']}] {line}")
    print(f"\nDECISION: {decision['action'].upper()} {decision['model']} -- {decision['reason']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Champion vs. challenger fairness comparison.")
    parser.add_argument("--dataset", choices=ingest.DATASET_NAMES, default=DEFAULT_DATASET)
    parser.add_argument("--champion", choices=list(MODELS), default=None,
                        help="override the champion from the thresholds file")
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH)
    args = parser.parse_args(argv)
    try:
        rows, decision = compare(args.dataset, args.thresholds, args.champion)
    except GateConfigError as e:
        print(f"ERROR {e}", file=sys.stderr)
        return 2

    out_dir = write_outputs(rows, decision, args.dataset)
    print_table(rows, decision)
    print(f"wrote {out_dir / 'comparison.json'} and comparison.csv")
    champ = next(r for r in rows if r["role"] == "champion")
    if not champ["gate_passed"]:
        print(f"fairness gate FAILED for champion {champ['model']}", file=sys.stderr)
        return 1
    print(f"fairness gate passed for champion {champ['model']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
