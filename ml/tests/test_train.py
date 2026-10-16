import math

import pandas as pd
import pytest

from ml import ingest, ingest_hmda, train
from ml.models import MODELS

pytestmark = pytest.mark.skipif(
    not ingest_hmda.DEFAULT_PATH.exists(), reason="HMDA file not downloaded"
)

PROTECTED = ["race", "ethnicity", "sex", "age_group"]


@pytest.fixture(scope="module")
def data():
    return ingest.load_dataset("hmda")


@pytest.fixture(scope="module")
def baseline(data):
    return train.train(data)[1]


def test_protected_attributes_not_used_as_features(data):
    assert set(data.features).isdisjoint({*PROTECTED, "approved"})


def test_split_is_stratified_and_disjoint(data):
    train_df, test_df = train.split(data)
    assert len(test_df) == math.ceil(0.3 * len(data.frame))  # sklearn rounds the test share up
    assert set(train_df.index).isdisjoint(test_df.index)
    assert test_df["approved"].mean() == pytest.approx(data.frame["approved"].mean(), abs=0.001)


def test_same_seed_gives_identical_predictions(data, baseline):
    _, again = train.train(data)
    pd.testing.assert_frame_equal(baseline, again)


def test_predictions_schema(data, baseline):
    assert list(baseline.columns) == ["row_id", "y_true", "y_pred", "y_prob", *PROTECTED]
    assert baseline["y_prob"].between(0, 1).all()
    assert set(baseline["y_pred"]) <= {0, 1}
    assert (baseline["y_pred"] == (baseline["y_prob"] >= 0.5)).all()
    # protected columns line up with the source rows
    src = data.frame.loc[baseline["row_id"]]
    assert (baseline["y_true"].to_numpy() == src["approved"].astype(int).to_numpy()).all()
    assert baseline["race"].isna().sum() == src["race"].isna().sum()


def test_beats_majority_baseline(data, baseline):
    majority = max(data.frame["approved"].mean(), 1 - data.frame["approved"].mean())
    assert train.summarize(baseline)["accuracy"] > majority


@pytest.mark.parametrize("model", list(MODELS))
def test_every_model_scores_the_same_test_rows(data, baseline, model):
    _, preds = train.train(data, model)
    assert preds["row_id"].tolist() == baseline["row_id"].tolist()
    assert preds["y_prob"].between(0, 1).all()


def test_run_writes_csv(tmp_path, data, baseline):
    out = tmp_path / "reports" / "predictions.csv"
    train.run(data, "vendor_a_logreg", out)
    assert len(pd.read_csv(out)) == len(baseline)


def test_default_output_path():
    path = train.predictions_path("hmda", "vendor_b_gbm")
    assert path.parts[-3:] == ("hmda", "vendor_b_gbm", "predictions.csv")
