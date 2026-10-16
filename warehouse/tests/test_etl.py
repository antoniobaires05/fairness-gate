"""ETL tests.

Transform tests use small hand-written report files and need no database.
The load test runs against a real SQL Server only when
FAIRNESS_TEST_SQL_CONNECTION_STRING is set (e.g. LocalDB on a dev machine); it is
skipped in CI, which loads the real Azure database instead.
"""

import json
import os

import pytest

from warehouse import etl

RUN_ID = "test-run-0001"


def write_build(tmp_path, run_id=RUN_ID, champion_passes=False):
    ds = tmp_path / "hmda"
    models = [("vendor_a_logreg", "champion", champion_passes),
              ("vendor_a_mitigated", "challenger", True)]
    comparison = {
        "run": {"run_id": run_id, "created_at": "2026-10-16T01:02:03+00:00", "dataset": "hmda",
                "champion": "vendor_a_logreg", "build_passed": champion_passes,
                "commit_sha": "abc123", "branch": "main", "trigger": "push"},
        "decision": {"action": "promote", "model": "vendor_a_mitigated", "reason": "test"},
        "models": [{"model": m, "role": role, "description": f"desc {m}", "n": 10,
                    "accuracy": 0.9, "roc_auc": 0.8, "approval_rate": 0.7,
                    "gate_passed": passed, "n_failures": 0 if passed else 1,
                    "recommended": m == "vendor_a_mitigated"} for m, role, passed in models],
    }
    (ds).mkdir(parents=True)
    (ds / "comparison.json").write_text(json.dumps(comparison))
    for m, _, passed in models:
        (ds / m).mkdir()
        records = [
            {"run_id": run_id, "attribute": "race", "group": "White", "metric": "selection_rate",
             "value": 0.9, "ci_lower": 0.85, "ci_upper": 0.94, "gated": False, "threshold": None,
             "passed": None, "n": 50, "n_excluded_missing": 3, "gate_eligible": True},
            {"run_id": run_id, "attribute": "race", "group": "Asian", "metric": "precision",
             "value": 0.0, "ci_lower": None, "ci_upper": None, "gated": False, "threshold": None,
             "passed": None, "n": 4, "n_excluded_missing": 3, "gate_eligible": False},
            {"run_id": run_id, "attribute": "race", "group": "all",
             "metric": "equalized_odds_difference", "value": 0.05 if passed else 0.2,
             "ci_lower": 0.01, "ci_upper": 0.3, "gated": True, "threshold": 0.1,
             "passed": passed, "n": 54, "n_excluded_missing": 3},
        ]
        (ds / m / "fairness_report.json").write_text(json.dumps(records))
    return tmp_path


# --- extract / transform (no database) ----------------------------------------


def test_extract_builds_dimension_and_fact_rows(tmp_path):
    load = etl.extract("hmda", write_build(tmp_path))
    assert load.run["run_id"] == RUN_ID
    assert [m["ModelName"] for m in load.models] == ["vendor_a_logreg", "vendor_a_mitigated"]
    assert load.groups == {("race", "White"), ("race", "Asian"), ("race", "all")}
    assert load.metrics == {"selection_rate": False, "precision": False,
                            "equalized_odds_difference": True}
    assert len(load.audit) == 6 and len(load.evaluations) == 2


def test_missing_values_become_null(tmp_path):
    load = etl.extract("hmda", write_build(tmp_path))
    precision = next(a for a in load.audit if a["MetricName"] == "precision")
    assert precision["CiLower"] is None and precision["CiUpper"] is None
    assert etl._num(float("nan")) is None


def test_vendor_names():
    assert etl.vendor_of("vendor_b_gbm") == "Vendor B"
    assert etl.vendor_of("custom") == "custom"


def test_date_key():
    from datetime import datetime
    assert etl.date_key(datetime(2026, 10, 16)) == 20261016


def test_report_from_another_run_rejected(tmp_path):
    root = write_build(tmp_path)
    path = root / "hmda" / "vendor_a_logreg" / "fairness_report.json"
    path.write_text(path.read_text().replace(RUN_ID, "some-other-run"))
    with pytest.raises(etl.EtlError, match="some-other-run"):
        etl.extract("hmda", root)


def test_missing_comparison_explains_what_to_run(tmp_path):
    with pytest.raises(etl.EtlError, match="ml.compare"):
        etl.extract("hmda", tmp_path)


def test_main_requires_connection_string(monkeypatch, capsys):
    monkeypatch.delenv(etl.CONNECTION_ENV, raising=False)
    assert etl.main(["--dataset", "hmda"]) == 2
    assert etl.CONNECTION_ENV in capsys.readouterr().err


def test_schema_batches_split_on_go():
    text = etl.SCHEMA_PATH.read_text()
    assert "FactFairnessAudit" in text and "CiLower" in text and "CiUpper" in text


# --- load (real SQL Server, opt-in) --------------------------------------------

TEST_DB = os.environ.get("FAIRNESS_TEST_SQL_CONNECTION_STRING")


@pytest.mark.skipif(not TEST_DB, reason="set FAIRNESS_TEST_SQL_CONNECTION_STRING to run")
def test_load_is_idempotent_and_dimension_first(tmp_path):
    conn = etl.connect(TEST_DB, attempts=1)
    try:
        etl.apply_schema(conn)
        cur = conn.cursor()

        def count(table, where="1=1", params=()):
            cur.execute(f"SELECT COUNT(*) FROM dbo.{table} t JOIN dbo.DimRun r ON r.RunKey = t.RunKey "
                        f"WHERE r.RunId = ? AND {where}", [RUN_ID, *params])
            return cur.fetchone()[0]

        load = etl.extract("hmda", write_build(tmp_path))
        etl.load_build(conn, load)
        etl.load_build(conn, load)  # re-run: replaces, never duplicates
        assert count("FactFairnessAudit") == 6
        assert count("FactModelEvaluation") == 2
        assert count("FactModelEvaluation", "t.Recommended = 1") == 1

        # SCD type 1: a new description overwrites the old one
        load.models[0]["Description"] = "updated description"
        etl.load_build(conn, load)
        cur.execute("SELECT Description FROM dbo.DimModel WHERE ModelName = 'vendor_a_logreg'")
        assert cur.fetchone()[0] == "updated description"
    finally:
        cur = conn.cursor()
        for table in ("FactFairnessAudit", "FactModelEvaluation"):
            cur.execute(f"DELETE t FROM dbo.{table} t JOIN dbo.DimRun r ON r.RunKey = t.RunKey "
                        "WHERE r.RunId = ?", [RUN_ID])
        cur.execute("DELETE FROM dbo.DimRun WHERE RunId = ?", [RUN_ID])
        conn.commit()
        conn.close()


# --- connection errors -------------------------------------------------------


@pytest.mark.parametrize("cs", [
    "Server=x;Uid=u;Pwd=s3cr3t!;Database=d;",
    "Server=x;Uid=u;Password=s3cr3t!;",
    "Server=x;Uid=u;PWD={s3cr3t!;x};",
])
def test_redact_removes_password(cs):
    assert "s3cr3t" not in etl.redact(f"failed for {cs}", cs)


def test_malformed_connection_string_fails_fast_without_leaking_password():
    cs = "Server=tcp:nosuchserver.invalid,1433;Uid=u;Pwd=s3cr3t!;Connection Timeout=60;"
    with pytest.raises(etl.EtlError) as err:
        etl.connect(cs, attempts=4, wait_seconds=60)  # would take minutes if it retried
    assert "connection timeout" in str(err.value).lower()
    assert "s3cr3t" not in str(err.value)
