import json

import pandas as pd
import pytest

from ml import fairness_eval as fe

SEX_ONLY = {
    "attributes": ["sex"],
    "gated_metrics": {"demographic_parity_difference": 0.10, "equalized_odds_difference": 0.10},
}


def thresholds_yaml(tmp_path, dpd=0.10, eod=0.10, attributes=("sex",)):
    path = tmp_path / "thresholds.yaml"
    attrs = "\n".join(f"  - {a}" for a in attributes)
    path.write_text(
        f"attributes:\n{attrs}\n"
        "gated_metrics:\n"
        f"  demographic_parity_difference:\n    max: {dpd}\n"
        f"  equalized_odds_difference:\n    max: {eod}\n"
    )
    return path


def biased_preds():
    """Group 'a' is always approved; group 'b' half the time. One row has unknown sex.

    Selection rate: a = 1.0, b = 0.5  -> demographic parity difference 0.5.
    TPR: a = 1.0, b = 0.5; FPR: a = 1.0, b = 0.5 -> equalized odds difference 0.5.
    """
    return pd.DataFrame({
        "y_true": [1, 1, 0, 0, 1, 1, 0, 0, 1],
        "y_pred": [1, 1, 1, 1, 1, 0, 1, 0, 0],
        "sex": ["a", "a", "a", "a", "b", "b", "b", "b", None],
    })


def fair_preds():
    """Both groups get identical treatment -> every gated metric is 0."""
    return pd.DataFrame({
        "y_true": [1, 0, 1, 0],
        "y_pred": [1, 0, 1, 0],
        "sex": ["a", "a", "b", "b"],
    })


def run(tmp_path, preds, **threshold_kwargs):
    preds_path = tmp_path / "preds.csv"
    preds.to_csv(preds_path, index=False)
    report = tmp_path / "report.json"
    code = fe.main([
        "--predictions", str(preds_path),
        "--thresholds", str(thresholds_yaml(tmp_path, **threshold_kwargs)),
        "--report", str(report),
    ])
    return code, report


# --- thresholds loading ---------------------------------------------------


def test_load_thresholds_reads_repo_file():
    config = fe.load_thresholds()
    assert config["attributes"] == ["sex", "age_group"]
    assert set(config["gated_metrics"]) == set(fe.GATED_METRICS)


@pytest.mark.parametrize(
    "text, message",
    [
        ("attributes: []\ngated_metrics: {demographic_parity_difference: {max: 0.1}}", "attributes"),
        ("attributes: [sex]\ngated_metrics: {demographic_parity_diff: {max: 0.1}}", "unknown gated metric"),
        ("attributes: [sex]\ngated_metrics: {demographic_parity_difference: {max: 1.5}}", "between 0 and 1"),
        ("attributes: [sex]\ngated_metrics: {demographic_parity_difference: {max: 'x'}}", "between 0 and 1"),
        ("attributes: [sex]\ngated_metrics: {demographic_parity_difference: 0.1}", "between 0 and 1"),
        ("- just\n- a list", "mapping"),
    ],
)
def test_bad_thresholds_rejected(tmp_path, text, message):
    path = tmp_path / "t.yaml"
    path.write_text(text)
    with pytest.raises(fe.GateConfigError, match=message):
        fe.load_thresholds(path)


# --- metric computation ---------------------------------------------------


def test_gated_values_match_hand_computation():
    records, failures = fe.evaluate(biased_preds(), SEX_ONLY, run_id="t")
    gated = {r["metric"]: r for r in records if r["gated"] and r["attribute"] == "sex"}
    assert gated["demographic_parity_difference"]["value"] == pytest.approx(0.5)
    assert gated["equalized_odds_difference"]["value"] == pytest.approx(0.5)
    assert {f["metric"] for f in failures if f["attribute"] == "sex"} == set(fe.GATED_METRICS)


def test_failure_names_the_groups():
    _, failures = fe.evaluate(biased_preds(), SEX_ONLY, run_id="t")
    dpd = next(f for f in failures if f["metric"] == "demographic_parity_difference")
    assert (dpd["rate"], dpd["low_group"], dpd["high_group"]) == ("selection_rate", "b", "a")
    assert dpd["low_value"] == pytest.approx(0.5)
    assert dpd["high_value"] == pytest.approx(1.0)


def test_recorded_metrics_per_group():
    records, _ = fe.evaluate(biased_preds(), SEX_ONLY, run_id="t")
    rec = {(r["group"], r["metric"]): r["value"] for r in records if not r["gated"]}
    assert rec[("a", "selection_rate")] == pytest.approx(1.0)
    assert rec[("b", "selection_rate")] == pytest.approx(0.5)
    assert rec[("a", "accuracy")] == pytest.approx(0.5)
    assert rec[("b", "recall")] == pytest.approx(0.5)
    assert rec[("a", "count")] == 4


def test_missing_attribute_rows_excluded():
    records, _ = fe.evaluate(biased_preds(), SEX_ONLY, run_id="t")
    sex_records = [r for r in records if r["attribute"] == "sex"]
    assert {r["group"] for r in sex_records} == {"a", "b", "all"}
    assert all(r["n_excluded_missing"] == 1 for r in sex_records)


def test_unknown_attribute_is_config_error():
    preds = fair_preds()
    with pytest.raises(fe.GateConfigError, match="age_group"):
        fe.evaluate(preds, fe.load_thresholds(), run_id="t")


# --- CLI (pass/fail on fixture files is covered in test_gate.py) -----------


def test_threshold_is_inclusive(tmp_path):
    code, _ = run(tmp_path, biased_preds(), dpd=0.5, eod=0.5)
    assert code == 0


def test_exit_2_on_missing_predictions(tmp_path, capsys):
    code = fe.main(["--predictions", str(tmp_path / "nope.csv"),
                    "--thresholds", str(thresholds_yaml(tmp_path))])
    assert code == 2
    assert "predictions not found" in capsys.readouterr().err


def test_report_is_one_record_per_metric_group_run(tmp_path):
    _, report = run(tmp_path, biased_preds())
    records = json.loads(report.read_text())
    keys = [(r["run_id"], r["attribute"], r["group"], r["metric"]) for r in records]
    assert len(keys) == len(set(keys))
    # 2 groups x 5 recorded metrics + 2 gated metrics
    assert len(records) == 2 * len(fe.RECORDED_METRICS) + len(fe.GATED_METRICS)
    assert len({r["run_id"] for r in records}) == 1
