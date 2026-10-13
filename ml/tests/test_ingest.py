import pandas as pd
import pytest

from ml import ingest

HEADER = " ".join(ingest.COLUMN_NAMES)
# One valid row: famges (9th) = 3 -> male, alter (13th) = 30, kredit (last) = 1.
VALID_ROW = "1 18 4 2 1049 1 2 4 3 1 4 2 30 3 1 1 3 2 1 2 1"


def write_asc(tmp_path, rows):
    path = tmp_path / "sample.asc"
    path.write_text("\n".join([HEADER, *rows]) + "\n")
    return path


def row_with(**overrides):
    values = dict(zip(ingest.COLUMN_NAMES, VALID_ROW.split()))
    values.update({k: str(v) for k, v in overrides.items()})
    return " ".join(values.values())


# --- unit tests on small synthetic files ---------------------------------


def test_columns_renamed_to_english(tmp_path):
    df = ingest.load(write_asc(tmp_path, [VALID_ROW]))
    assert list(df.columns[:21]) == list(ingest.COLUMN_NAMES.values())
    assert {"approved", "sex", "age_group"} <= set(df.columns)


@pytest.mark.parametrize("kredit, expected", [(1, 1), (0, 0)])
def test_approved_from_credit_risk(tmp_path, kredit, expected):
    df = ingest.load(write_asc(tmp_path, [row_with(kredit=kredit)]))
    assert pd.api.types.is_integer_dtype(df["approved"])
    assert df["approved"].iloc[0] == expected


@pytest.mark.parametrize(
    "famges, expected", [(1, "male"), (3, "male"), (4, "female"), (2, pd.NA)]
)
def test_sex_mapping(tmp_path, famges, expected):
    df = ingest.load(write_asc(tmp_path, [row_with(famges=famges)]))
    value = df["sex"].iloc[0]
    if expected is pd.NA:
        assert pd.isna(value)
    else:
        assert value == expected


@pytest.mark.parametrize(
    "age, expected",
    [(19, "<25"), (24, "<25"), (25, "25+"), (75, "25+")],
)
def test_age_group_boundaries(tmp_path, age, expected):
    df = ingest.load(write_asc(tmp_path, [row_with(alter=age)]))
    assert df["age_group"].iloc[0] == expected


def test_age_group_is_ordered_categorical(tmp_path):
    df = ingest.load(write_asc(tmp_path, [VALID_ROW]))
    assert isinstance(df["age_group"].dtype, pd.CategoricalDtype)
    assert df["age_group"].cat.ordered
    assert list(df["age_group"].cat.categories) == ingest.AGE_LABELS


def test_does_not_mutate_input():
    raw = pd.DataFrame({"credit_risk": [1], "personal_status_sex": [3], "age": [30]})
    before = raw.copy()
    ingest.add_derived_columns(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_rejects_out_of_range_code(tmp_path):
    with pytest.raises(ValueError, match="personal_status_sex"):
        ingest.load(write_asc(tmp_path, [row_with(famges=9)]))


def test_rejects_missing_column(tmp_path):
    path = tmp_path / "bad.asc"
    path.write_text(HEADER.replace("kredit", "other") + "\n" + VALID_ROW + "\n")
    with pytest.raises(ValueError, match="kredit"):
        ingest.load(path)


def test_rejects_non_positive_age(tmp_path):
    with pytest.raises(ValueError, match="age"):
        ingest.load(write_asc(tmp_path, [row_with(alter=0)]))


# --- integration tests on the real file -----------------------------------

real_file = pytest.mark.skipif(
    not ingest.DEFAULT_PATH.exists(), reason="data/raw/SouthGermanCredit.asc not present"
)


@real_file
def test_real_file_shape_and_targets():
    df = ingest.load()
    assert len(df) == 1000
    assert df["approved"].sum() == 700


@real_file
def test_real_file_sex_counts():
    sex = ingest.load()["sex"]
    assert (sex == "male").sum() == 598  # codes 1 (50) + 3 (548)
    assert (sex == "female").sum() == 92  # code 4
    assert sex.isna().sum() == 310  # code 2, ambiguous


@real_file
def test_real_file_no_missing_age_group():
    df = ingest.load()
    assert df["age_group"].notna().all()
    assert df["age"].between(19, 75).all()
