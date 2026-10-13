import pandas as pd
import pytest

from ml import ingest, train

pytestmark = pytest.mark.skipif(
    not ingest.DEFAULT_PATH.exists(), reason="data/raw/SouthGermanCredit.asc not present"
)


@pytest.fixture(scope="module")
def data():
    return ingest.load()


def test_protected_attributes_not_used_as_features():
    features = set(train.NUMERIC_FEATURES + train.CATEGORICAL_FEATURES)
    assert features.isdisjoint(
        {"sex", "age_group", "age", "personal_status_sex", "approved", "credit_risk"}
    )


def test_split_is_stratified_and_disjoint(data):
    train_df, test_df = train.split(data)
    assert len(test_df) == 300
    assert set(train_df.index).isdisjoint(test_df.index)
    assert test_df["approved"].mean() == pytest.approx(0.7)


def test_same_seed_gives_identical_predictions(data):
    _, a = train.train(data)
    _, b = train.train(data)
    pd.testing.assert_frame_equal(a, b)


def test_predictions_schema(data):
    _, preds = train.train(data)
    assert list(preds.columns) == ["row_id", "y_true", "y_pred", "y_prob", "sex", "age_group"]
    assert preds["y_prob"].between(0, 1).all()
    assert set(preds["y_pred"]) <= {0, 1}
    assert (preds["y_pred"] == (preds["y_prob"] >= 0.5)).all()
    # protected columns line up with the source rows
    src = data.loc[preds["row_id"]]
    assert (preds["y_true"].to_numpy() == src["approved"].astype(int).to_numpy()).all()
    assert preds["sex"].isna().sum() == src["sex"].isna().sum()


def test_beats_majority_baseline(data):
    _, preds = train.train(data)
    accuracy = (preds["y_true"] == preds["y_pred"]).mean()
    assert accuracy > 0.70


def test_main_writes_csv(tmp_path):
    out = tmp_path / "reports" / "predictions.csv"
    train.main(out)
    written = pd.read_csv(out)
    assert len(written) == 300
