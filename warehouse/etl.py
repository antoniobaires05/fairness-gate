"""Load one build's fairness audit into the star schema (Azure SQL / SQL Server).

Usage:
    python -m warehouse.etl [--dataset hmda] [--init-schema]

Reads ml/reports/<dataset>/comparison.json (from ml.compare) and each model's
fairness_report.json, then:
  1. extract  -- read the JSON files
  2. transform -- turn them into dimension rows and fact rows keyed by natural keys
  3. load      -- upsert dimensions first (MERGE on natural key), look up surrogate
                  keys, then replace this build's fact rows in one transaction.
Re-running for the same build replaces its rows instead of duplicating them.

The connection string comes from the FAIRNESS_SQL_CONNECTION_STRING environment
variable (a GitHub Secret in CI) and is never written to disk or logs. Example for
local SQL Server LocalDB:
    Server=(localdb)\\MSSQLLocalDB;Database=FairnessGates;Trusted_Connection=yes;Encrypt=no;
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "ml" / "reports"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
CONNECTION_ENV = "FAIRNESS_SQL_CONNECTION_STRING"

GATED = {"demographic_parity_difference", "equalized_odds_difference"}


class EtlError(RuntimeError):
    pass


# --- Extract + transform ---------------------------------------------------------


@dataclass
class BuildLoad:
    """One build, as rows keyed by natural keys (no database needed to build this)."""

    run: dict
    dataset: str
    models: list[dict] = field(default_factory=list)       # ModelName, Vendor, Description
    groups: set[tuple[str, str]] = field(default_factory=set)  # (attribute, group)
    metrics: dict[str, bool] = field(default_factory=dict)     # MetricName -> IsGated
    audit: list[dict] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)


def _num(value):
    """JSON numbers -> float, with NaN/None -> None (SQL NULL)."""
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def vendor_of(model_name: str) -> str:
    m = re.match(r"vendor_([a-z])", model_name)
    return f"Vendor {m.group(1).upper()}" if m else model_name


def date_key(ts: datetime) -> int:
    return ts.year * 10000 + ts.month * 100 + ts.day


def extract(dataset: str, reports_dir: Path = REPORTS_DIR) -> BuildLoad:
    comparison_path = reports_dir / dataset / "comparison.json"
    try:
        comparison = json.loads(comparison_path.read_text())
    except FileNotFoundError:
        raise EtlError(f"{comparison_path} not found (run python -m ml.compare --dataset {dataset})") from None
    if "run" not in comparison:
        raise EtlError(f"{comparison_path} has no 'run' block; re-run python -m ml.compare")

    load = BuildLoad(run=comparison["run"], dataset=dataset)
    for model in comparison["models"]:
        name = model["model"]
        load.models.append({"ModelName": name, "Vendor": vendor_of(name),
                            "Description": model["description"]})
        load.evaluations.append({
            "ModelName": name, "Role": model["role"], "TestRows": model["n"],
            "Accuracy": model["accuracy"], "RocAuc": model["roc_auc"],
            "ApprovalRate": model["approval_rate"], "GatePassed": model["gate_passed"],
            "GateFailures": model["n_failures"], "Recommended": model.get("recommended", False),
        })
        report_path = reports_dir / dataset / name / "fairness_report.json"
        try:
            records = json.loads(report_path.read_text())
        except FileNotFoundError:
            raise EtlError(f"{report_path} not found") from None
        for r in records:
            if r["run_id"] != load.run["run_id"]:
                raise EtlError(f"{report_path} is from run {r['run_id']}, not {load.run['run_id']}")
            load.groups.add((r["attribute"], r["group"]))
            load.metrics[r["metric"]] = r["metric"] in GATED
            load.audit.append({
                "ModelName": name, "AttributeName": r["attribute"], "GroupName": r["group"],
                "MetricName": r["metric"], "MetricValue": _num(r["value"]),
                "CiLower": _num(r.get("ci_lower")), "CiUpper": _num(r.get("ci_upper")),
                "Threshold": _num(r.get("threshold")), "Passed": r.get("passed"),
                "GateEligible": r.get("gate_eligible"), "GroupSize": int(r.get("n") or 0),
                "RowsMissingAttr": int(r.get("n_excluded_missing") or 0),
            })
    return load


# --- Load ----------------------------------------------------------------------------


def connect(connection_string: str, attempts: int = 4, wait_seconds: float = 20.0):
    """Connect, retrying: a paused Azure SQL free/serverless database takes ~1 min to wake."""
    import time

    import mssql_python  # imported here so extract/transform work without the driver

    for attempt in range(1, attempts + 1):
        try:
            return mssql_python.connect(connection_string)
        except Exception as e:  # driver raises its own error types
            if attempt == attempts:
                raise EtlError(f"could not connect after {attempts} attempts: {type(e).__name__}") from None
            print(f"connect attempt {attempt} failed ({type(e).__name__}); retrying in {wait_seconds:.0f}s")
            time.sleep(wait_seconds)


def apply_schema(conn, schema_path: Path = SCHEMA_PATH) -> None:
    """Run schema.sql batch by batch (batches are separated by GO lines)."""
    batches = re.split(r"^\s*GO\s*$", schema_path.read_text(), flags=re.MULTILINE | re.IGNORECASE)
    cur = conn.cursor()
    for batch in batches:
        if batch.strip():
            cur.execute(batch)
    conn.commit()


def _merge(cur, table: str, natural: dict, other: dict | None = None) -> None:
    """Insert a dimension row, or overwrite its non-key columns (SCD type 1)."""
    other = other or {}
    cols = {**natural, **other}
    source = ", ".join(f"? AS [{c}]" for c in cols)
    on = " AND ".join(f"t.[{c}] = s.[{c}]" for c in natural)
    insert_cols = ", ".join(f"[{c}]" for c in cols)
    insert_vals = ", ".join(f"s.[{c}]" for c in cols)
    update = (" WHEN MATCHED THEN UPDATE SET " + ", ".join(f"t.[{c}] = s.[{c}]" for c in other)
              if other else "")
    cur.execute(
        f"MERGE dbo.{table} AS t USING (SELECT {source}) AS s ON {on}{update} "
        f"WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals});",
        list(cols.values()),
    )


def _keys(cur, table: str, key: str, natural: list[str]) -> dict:
    cur.execute(f"SELECT [{key}], {', '.join(f'[{c}]' for c in natural)} FROM dbo.{table}")
    rows = cur.fetchall()
    return {(tuple(r[1:]) if len(natural) > 1 else r[1]): r[0] for r in rows}


def load_build(conn, load: BuildLoad) -> dict:
    """Dimensions first, then this build's facts (replaced atomically). Returns row counts."""
    run = load.run
    started = datetime.fromisoformat(run["created_at"])
    dkey = date_key(started)
    cur = conn.cursor()
    try:
        # 1. Dimensions
        _merge(cur, "DimDate", {"DateKey": dkey}, {
            "FullDate": started.date(), "Year": started.year,
            "Quarter": (started.month - 1) // 3 + 1, "Month": started.month,
            "MonthName": started.strftime("%B"), "Day": started.day,
            "DayName": started.strftime("%A"),
        })
        _merge(cur, "DimRun", {"RunId": run["run_id"]}, {
            "RunStartedAt": started.replace(tzinfo=None), "DateKey": dkey,
            "CommitSha": run.get("commit_sha"), "Branch": run.get("branch"),
            "Trigger": run.get("trigger") or "local", "ChampionModel": run["champion"],
            "BuildPassed": bool(run["build_passed"]),
        })
        _merge(cur, "DimDataset", {"DatasetName": load.dataset})
        for m in load.models:
            _merge(cur, "DimModel", {"ModelName": m["ModelName"]},
                   {"Vendor": m["Vendor"], "Description": m["Description"]})
        for attribute, group in sorted(load.groups):
            _merge(cur, "DimGroup", {"AttributeName": attribute, "GroupName": group})
        for metric, gated in sorted(load.metrics.items()):
            _merge(cur, "DimMetric", {"MetricName": metric},
                   {"IsGated": gated, "MetricLevel": "gap" if gated else "group"})

        # 2. Surrogate key lookups
        run_key = _keys(cur, "DimRun", "RunKey", ["RunId"])[run["run_id"]]
        dataset_key = _keys(cur, "DimDataset", "DatasetKey", ["DatasetName"])[load.dataset]
        model_keys = _keys(cur, "DimModel", "ModelKey", ["ModelName"])
        group_keys = _keys(cur, "DimGroup", "GroupKey", ["AttributeName", "GroupName"])
        metric_keys = _keys(cur, "DimMetric", "MetricKey", ["MetricName"])

        # 3. Facts: replace this build's rows so a re-run never duplicates them
        for table in ("FactFairnessAudit", "FactModelEvaluation"):
            cur.execute(f"DELETE FROM dbo.{table} WHERE RunKey = ? AND DatasetKey = ?",
                        [run_key, dataset_key])
        audit_rows = [
            (run_key, dkey, dataset_key, model_keys[a["ModelName"]],
             group_keys[(a["AttributeName"], a["GroupName"])], metric_keys[a["MetricName"]],
             a["MetricValue"], a["CiLower"], a["CiUpper"], a["Threshold"], a["Passed"],
             a["GateEligible"], a["GroupSize"], a["RowsMissingAttr"])
            for a in load.audit
        ]
        cur.executemany(
            "INSERT INTO dbo.FactFairnessAudit (RunKey, DateKey, DatasetKey, ModelKey, GroupKey, "
            "MetricKey, MetricValue, CiLower, CiUpper, Threshold, Passed, GateEligible, GroupSize, "
            "RowsMissingAttr) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            audit_rows,
        )
        eval_rows = [
            (run_key, dkey, dataset_key, model_keys[e["ModelName"]], e["Role"], e["TestRows"],
             e["Accuracy"], e["RocAuc"], e["ApprovalRate"], bool(e["GatePassed"]),
             e["GateFailures"], bool(e["Recommended"]))
            for e in load.evaluations
        ]
        cur.executemany(
            "INSERT INTO dbo.FactModelEvaluation (RunKey, DateKey, DatasetKey, ModelKey, [Role], "
            "TestRows, Accuracy, RocAuc, ApprovalRate, GatePassed, GateFailures, Recommended) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            eval_rows,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return {"FactFairnessAudit": len(audit_rows), "FactModelEvaluation": len(eval_rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load a build's fairness audit into the warehouse.")
    parser.add_argument("--dataset", default="hmda")
    parser.add_argument("--init-schema", action="store_true", help="create tables/views if missing")
    args = parser.parse_args(argv)

    connection_string = os.environ.get(CONNECTION_ENV)
    if not connection_string:
        print(f"ERROR set {CONNECTION_ENV} to the database connection string", file=sys.stderr)
        return 2
    try:
        load = extract(args.dataset)
        conn = connect(connection_string)
        try:
            if args.init_schema:
                apply_schema(conn)
            counts = load_build(conn, load)
        finally:
            conn.close()
    except EtlError as e:
        print(f"ERROR {e}", file=sys.stderr)
        return 2
    print(f"loaded run {load.run['run_id']} ({args.dataset}): "
          + ", ".join(f"{n} rows into {t}" for t, n in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
