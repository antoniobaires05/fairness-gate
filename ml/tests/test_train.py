import pandas as pd
import pytest

from ml import ingest, train
from ml.models import MODELS

pytestmark = pytest.mark.skipif(
    not ingest.DEFAULT_PATH.exists(), reason="data/raw/SouthGermanCredit.asc not present"
)


@pytest.fixture(scope="module")
def data():
    return ingest.load_dataset("german_credit")


def test_protected_attributes_not_used_as_features(data):
    assert set(data.features).isdisjoint(
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
    src = data.frame.loc[preds["row_id"]]
    assert (preds["y_true"].to_numpy() == src["approved"].astype(int).to_numpy()).all()
    assert preds["sex"].isna().sum() == src["sex"].isna().sum()


def test_beats_majority_baseline(data):
    _, preds = train.train(data)
    assert train.summarize(preds)["accuracy"] > 0.70


@pytest.mark.parametrize("model", list(MODELS))
def test_every_model_scores_the_same_test_rows(data, model):
    _, baseline = train.train(data)
    _, preds = train.train(data, model)
    assert preds["row_id"].tolist() == baseline["row_id"].tolist()
    assert preds["y_prob"].between(0, 1).all()


def test_run_writes_csv(tmp_path, data):
    out = tmp_path / "reports" / "predictions.csv"
    train.run(data, "vendor_a_logreg", out)
    assert len(pd.read_csv(out)) == 300


def test_default_output_path():
    path = train.predictions_path("hmda", "vendor_b_gbm")
    assert path.parts[-3:] == ("hmda", "vendor_b_gbm", "predictions.csv")
