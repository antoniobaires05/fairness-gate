"""Run the fairness gate as CI does (python -m ml.fairness_eval) and check its exit code.

Fixtures (ml/tests/fixtures/), 2 repeats of 4 rows per (sex, age_group) cell:
    fair_predictions.csv    every group: selection 0.75, TPR 1.0, FPR 0.5 -> all gaps 0
    biased_predictions.csv  female: selection 0.25, TPR 0.5, FPR 0.0
                            male:   selection 1.0,  TPR 1.0, FPR 1.0
                            -> sex DPD 0.75, EOD 1.0; age_group gaps stay under 0.10.
                            Plus 2 rows with missing sex.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def run_gate(predictions: Path, report: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ml.fairness_eval",
         "--predictions", str(predictions),
         "--thresholds", str(FIXTURES / "thresholds.yaml"),
         "--report", str(report)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


@pytest.fixture(scope="module")
def biased(tmp_path_factory):
    report = tmp_path_factory.mktemp("biased") / "report.json"
    return run_gate(FIXTURES / "biased_predictions.csv", report), report


@pytest.fixture(scope="module")
def fair(tmp_path_factory):
    report = tmp_path_factory.mktemp("fair") / "report.json"
    return run_gate(FIXTURES / "fair_predictions.csv", report), report


def test_biased_predictions_fail_the_gate(biased):
    result, _ = biased
    assert result.returncode == 1, result.stdout + result.stderr
    assert "fairness gate FAILED: 2 metric(s)" in result.stderr


def test_failure_names_metric_attribute_and_groups(biased):
    out = biased[0].stdout
    assert ("FAIL demographic_parity_difference on 'sex': 0.750 > max 0.100"
            " -- selection_rate female=0.250 vs male=1.000") in out
    assert ("FAIL equalized_odds_difference on 'sex': 1.000 > max 0.100"
            " -- false_positive_rate female=0.000 vs male=1.000") in out


def test_only_the_biased_attribute_fails(biased):
    out = biased[0].stdout
    assert "PASS demographic_parity_difference on 'age_group'" in out
    assert "PASS equalized_odds_difference on 'age_group'" in out
    assert "on 'age_group'" not in "\n".join(l for l in out.splitlines() if l.startswith("FAIL"))


def test_biased_report_marks_failures(biased):
    records = json.loads(biased[1].read_text())
    failed = {(r["attribute"], r["metric"]) for r in records if r["gated"] and not r["passed"]}
    assert failed == {("sex", "demographic_parity_difference"),
                      ("sex", "equalized_odds_difference")}
    assert all(r["n_excluded_missing"] == 2 for r in records if r["attribute"] == "sex")


def test_fair_predictions_pass_the_gate(fair):
    result, report = fair
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
    assert "fairness gate passed" in result.stdout
    gated = [r for r in json.loads(report.read_text()) if r["gated"]]
    assert len(gated) == 4
    assert all(r["passed"] and r["value"] == pytest.approx(0) for r in gated)
