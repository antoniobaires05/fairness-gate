import math

import pandas as pd
import pytest

from ml import ingest_hmda as hmda

# One in-scope, approved application with every column ingest_hmda reads.
BASE_ROW = {
    "action_taken": "1",
    "lien_status": "1",
    "derived_dwelling_category": "Single Family (1-4 Units):Site-Built",
    "occupancy_type": "1",
    "reverse_mortgage": "2",
    "open-end_line_of_credit": "2",
    "business_or_commercial_purpose": "2",
    "loan_purpose": "1",
    "derived_race": "White",
    "derived_ethnicity": "Not Hispanic or Latino",
    "derived_sex": "Male",
    "applicant_age_above_62": "No",
    "income": "100",
    "loan_amount": "250000",
    "property_value": "300000",
    "loan_to_value_ratio": "83.333",
    "debt_to_income_ratio": "36",
    "loan_term": "360",
    "loan_type": "1",
    "preapproval": "2",
    "conforming_loan_limit": "C",
    "co-applicant_age": "9999",
}


def raw(*overrides):
    return pd.DataFrame([{**BASE_ROW, **o} for o in overrides]).astype(object)


def clean(*overrides):
    frame, _ = hmda.clean(raw(*overrides))
    return frame


# --- Rows ---------------------------------------------------------------------


@pytest.mark.parametrize("action, kept, approved", [
    ("1", True, 1), ("2", True, 1), ("3", True, 0),
    ("4", False, None), ("5", False, None), ("6", False, None), ("7", False, None), ("8", False, None),
])
def test_action_taken_mapping(action, kept, approved):
    frame = clean({"action_taken": action})
    assert (len(frame) == 1) is kept
    if kept:
        assert frame["approved"].iloc[0] == approved


@pytest.mark.parametrize("column, value", [
    ("lien_status", "2"),
    ("derived_dwelling_category", "Single Family (1-4 Units):Manufactured"),
    ("occupancy_type", "3"),
    ("reverse_mortgage", "1"),
    ("open-end_line_of_credit", "1"),
    ("business_or_commercial_purpose", "1"),
    ("loan_purpose", "2"),
    ("reverse_mortgage", "1111"),  # exempt -> unknown -> out of scope
])
def test_out_of_scope_rows_dropped(column, value):
    assert len(clean({column: value})) == 0


def test_funnel_log_counts_each_step():
    _, log = hmda.clean(raw({}, {"action_taken": "4"}, {"lien_status": "2"}))
    counts = dict((step.split(":")[0], n) for step, n in log)
    assert counts["raw"] == 3
    assert counts["decision made (approved or denied)"] == 2
    assert log[-1][1] == 1


# --- Features -----------------------------------------------------------------


@pytest.mark.parametrize("dti, expected", [
    ("<20%", 15.0), ("20%-<30%", 25.0), ("30%-<36%", 33.0), ("42", 42.0),
    ("50%-60%", 55.0), (">60%", 65.0), ("Exempt", math.nan), (None, math.nan),
])
def test_debt_to_income_parsing(dti, expected):
    value = clean({"debt_to_income_ratio": dti})["debt_to_income_ratio"].iloc[0]
    assert (math.isnan(value) and math.isnan(expected)) or value == expected


@pytest.mark.parametrize("column, value, feature", [
    ("income", "-50", "log_income"),
    ("income", "0", "log_income"),
    ("loan_to_value_ratio", "Exempt", "loan_to_value_ratio"),
    ("loan_to_value_ratio", "999", "loan_to_value_ratio"),
    ("property_value", "Exempt", "log_property_value"),
    ("loan_term", "5", "loan_term"),
])
def test_invalid_numbers_become_missing(column, value, feature):
    assert pd.isna(clean({column: value})[feature].iloc[0])


def test_has_coapplicant():
    frame = clean({"co-applicant_age": "9999"}, {"co-applicant_age": "35-44"})
    assert frame["has_coapplicant"].tolist() == ["no", "yes"]


def test_features_exclude_protected_and_target():
    features = set(hmda.NUMERIC_FEATURES + hmda.CATEGORICAL_FEATURES)
    assert features.isdisjoint(hmda.PROTECTED + ["approved"])
    assert set(clean({}).columns) == features | set(hmda.PROTECTED) | {"approved"}


# --- Protected attributes -------------------------------------------------------


@pytest.mark.parametrize("column, value", [
    ("derived_race", "Race Not Available"),
    ("derived_race", "Free Form Text Only"),
    ("derived_ethnicity", "Ethnicity Not Available"),
    ("derived_sex", "Sex Not Available"),
])
def test_not_provided_becomes_missing(column, value):
    frame = clean({column: value})
    attribute = column.removeprefix("derived_")
    assert pd.isna(frame[attribute].iloc[0])


@pytest.mark.parametrize("above_62, expected", [("Yes", "62+"), ("No", "under 62"), ("NA", None)])
def test_age_group_from_reg_b_62_line(above_62, expected):
    value = clean({"applicant_age_above_62": above_62})["age_group"].iloc[0]
    assert (pd.isna(value) and expected is None) or value == expected


def test_real_groups_kept_as_reported():
    frame = clean({"derived_race": "Black or African American"}, {"derived_race": "Joint"})
    assert frame["race"].tolist() == ["Black or African American", "Joint"]


# --- Loading ------------------------------------------------------------------


def test_sampling_is_seeded(tmp_path):
    path = tmp_path / "lar.csv"
    raw(*[{"loan_amount": str(100_000 + i)} for i in range(50)]).to_csv(path, index=False)
    a = hmda.load_raw(path, sample_size=10, seed=1)
    b = hmda.load_raw(path, sample_size=10, seed=1)
    c = hmda.load_raw(path, sample_size=10, seed=2)
    assert len(a) == 10
    assert a["loan_amount"].tolist() == b["loan_amount"].tolist()
    assert a["loan_amount"].tolist() != c["loan_amount"].tolist()


def test_missing_required_column_rejected(tmp_path):
    path = tmp_path / "lar.csv"
    raw({}).drop(columns=["action_taken"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="action_taken"):
        hmda.load_raw(path)


@pytest.mark.skipif(not hmda.DEFAULT_PATH.exists(), reason="HMDA file not downloaded")
def test_real_file_funnel():
    ds = hmda.load_dataset()
    assert len(ds.frame) == 16_511
    assert ds.frame["approved"].mean() == pytest.approx(0.848, abs=0.001)
    assert ds.frame[ds.protected].notna().any().all()
