import json

import pytest

from ml import compare, ingest_hmda, train
from ml.fairness_eval import GateConfigError


def row(model, role, accuracy, passed):
    return {"model": model, "role": role, "accuracy": accuracy, "gate_passed": passed}


# --- decision rule -------------------------------------------------------------


def test_promotes_passing_challenger_when_champion_fails():
    rows = [row("champ", "champion", 0.87, False), row("fair", "challenger", 0.85, True)]
    decision = compare.decide(rows, "champ", max_accuracy_drop=0.05)
    assert (decision["action"], decision["model"]) == ("promote", "fair")


def test_keeps_champion_when_challenger_too_inaccurate():
    rows = [row("champ", "champion", 0.87, False), row("fair", "challenger", 0.70, True)]
    decision = compare.decide(rows, "champ", max_accuracy_drop=0.05)
    assert (decision["action"], decision["model"]) == ("keep", "champ")
    assert "escalate" in decision["reason"]


def test_never_promotes_a_failing_challenger():
    rows = [row("champ", "champion", 0.80, False), row("accurate", "challenger", 0.95, False)]
    assert compare.decide(rows, "champ", 0.05)["model"] == "champ"


def test_keeps_passing_champion_over_less_accurate_challenger():
    rows = [row("champ", "champion", 0.87, True), row("other", "challenger", 0.86, True)]
    assert compare.decide(rows, "champ", 0.05) ["action"] == "keep"


def test_promotes_more_accurate_passing_challenger():
    rows = [row("champ", "champion", 0.85, True), row("better", "challenger", 0.88, True),
            row("best_but_unfair", "challenger", 0.95, False)]
    assert compare.decide(rows, "champ", 0.05)["model"] == "better"


# --- policy config ---------------------------------------------------------------


def test_repo_policy_loads():
    policy = compare.load_policy(compare.THRESHOLDS_PATH, "hmda")
    assert policy["champion"] == "vendor_a_logreg"
    assert "vendor_a_mitigated" in policy["challengers"]


def test_unknown_model_in_policy_rejected(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("defaults: {champion: vendor_z, challengers: []}\ndatasets: {hmda: {}}\n")
    with pytest.raises(GateConfigError, match="vendor_z"):
        compare.load_policy(path, "hmda")


# --- end to end (real HMDA file) ---------------------------------------------------


@pytest.mark.skipif(not ingest_hmda.DEFAULT_PATH.exists(), reason="HMDA file not downloaded")
def test_exit_code_follows_champion(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "REPORTS_DIR", tmp_path)
    assert compare.main(["--dataset", "hmda"]) == 1  # vendor A fails on HMDA
    result = json.loads((tmp_path / "hmda" / "comparison.json").read_text())
    assert {m["model"] for m in result["models"]} == {
        "vendor_a_logreg", "vendor_b_gbm", "vendor_a_mitigated"}
    assert sum(m["role"] == "champion" for m in result["models"]) == 1
    assert (tmp_path / "hmda" / "comparison.csv").exists()
    # one build = one run id, shared by every model's report (the warehouse's DimRun key)
    run = result["run"]
    assert run["champion"] == "vendor_a_logreg" and run["build_passed"] is False
    for m in result["models"]:
        report = json.loads((tmp_path / "hmda" / m["model"] / "fairness_report.json").read_text())
        assert {r["run_id"] for r in report} == {run["run_id"]}
