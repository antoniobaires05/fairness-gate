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
    assert config["attributes"] == ["race", "ethnicity", "sex", "age_group"]
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
    with pytest.raises(fe.GateConfigError, match="race"):
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


# --- group-size rules, confidence intervals, sectioned config ----------------


def uneven_preds():
    """Groups a and b (40 rows each) are treated differently; c has only 3 rows."""
    return pd.DataFrame({
        "y_true": [1, 0] * 20 + [1, 0] * 20 + [1, 0, 1],
        "y_pred": [1] * 40 + [1, 0, 0, 0] * 10 + [0, 0, 0],
        "sex": ["a"] * 40 + ["b"] * 40 + ["c"] * 3,
    })


def gated(records, metric):
    return next(r for r in records if r["gated"] and r["metric"] == metric)


def test_small_groups_reported_but_not_gated():
    cfg = {**SEX_ONLY, "min_group_size": 10, "bootstrap_samples": 200}
    records, _ = fe.evaluate(uneven_preds(), cfg, run_id="t")
    dpd = gated(records, "demographic_parity_difference")
    assert dpd["groups_gated"] == ["a", "b"]
    assert dpd["groups_skipped"] == ["c"]
    assert dpd["value"] == pytest.approx(0.75)  # a=1.0 vs b=0.25; c (0.0) ignored
    assert any(r["group"] == "c" and not r["gate_eligible"] for r in records if not r["gated"])


def test_min_outcome_count_applies_to_equalized_odds_only():
    cfg = {**SEX_ONLY, "min_group_size": 10, "min_outcome_count": 25, "bootstrap_samples": 200}
    records, _ = fe.evaluate(uneven_preds(), cfg, run_id="t")
    assert gated(records, "demographic_parity_difference")["passed"] is False
    eod = gated(records, "equalized_odds_difference")  # each group has 20 of each outcome
    assert eod["passed"] is None and eod["value"] is None


def test_skip_printed_when_too_few_groups(capsys):
    cfg = {**SEX_ONLY, "min_group_size": 50}
    records, failures = fe.evaluate(uneven_preds(), cfg, run_id="t")
    fe.print_summary(records, failures)
    assert failures == []
    assert "SKIP demographic_parity_difference on 'sex'" in capsys.readouterr().out


def test_records_carry_confidence_intervals():
    records, _ = fe.evaluate(uneven_preds(), {**SEX_ONLY, "bootstrap_samples": 200}, run_id="t")
    for r in records:
        if r["metric"] == "count" or (r["group"] == "c" and r["metric"] == "precision"):
            assert r["ci_lower"] is None  # counts have no CI; c has no predicted approvals
        elif r["value"] is not None:
            assert r["ci_lower"] <= r["value"] + 1e-9 and r["value"] <= r["ci_upper"] + 1e-9
    sel_b = next(r for r in records if r["group"] == "b" and r["metric"] == "selection_rate")
    assert (sel_b["ci_lower"], sel_b["ci_upper"]) == pytest.approx(fe.wilson_interval(10, 40))


def test_gate_on_ci_upper_is_stricter():
    preds = uneven_preds()
    base = {**SEX_ONLY, "min_group_size": 10, "bootstrap_samples": 200,
            "gated_metrics": {"demographic_parity_difference": 0.76}}
    _, point = fe.evaluate(preds, base, run_id="t")
    _, upper = fe.evaluate(preds, {**base, "gate_on": "ci_upper"}, run_id="t")
    assert point == [] and len(upper) == 1


def test_records_name_dataset_and_model():
    records, _ = fe.evaluate(fair_preds(), SEX_ONLY, run_id="t", dataset="hmda", model="m1")
    assert {(r["dataset"], r["model"]) for r in records} == {("hmda", "m1")}


def test_sectioned_config_resolves_per_dataset():
    hmda = fe.load_thresholds(dataset="hmda")
    assert hmda["attributes"] == ["race", "ethnicity", "sex", "age_group"]
    assert hmda["min_group_size"] == 100 and hmda["min_outcome_count"] == 30
    assert set(hmda["gated_metrics"]) == {"demographic_parity_difference",
                                          "equalized_odds_difference"}


def test_unknown_dataset_section_rejected():
    with pytest.raises(fe.GateConfigError, match="no section"):
        fe.load_thresholds(dataset="nope")


@pytest.mark.parametrize("text, message", [
    ("attributes: [sex]\ngated_metrics: {demographic_parity_difference: {max: 0.1}}\ngate_on: maybe", "gate_on"),
    ("attributes: [sex]\ngated_metrics: {demographic_parity_difference: {max: 0.1}}\nmin_group_size: 0", "min_group_size"),
    ("attributes: [sex]\ngated_metrics: {demographic_parity_difference: {max: 0.1}}\nconfidence_level: 95", "confidence_level"),
])
def test_bad_settings_rejected(tmp_path, text, message):
    path = tmp_path / "t.yaml"
    path.write_text(text)
    with pytest.raises(fe.GateConfigError, match=message):
        fe.load_thresholds(path)
