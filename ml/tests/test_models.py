import numpy as np
import pandas as pd
import pytest

from ml import models
from ml.ingest import Dataset


@pytest.fixture(scope="module")
def dataset():
    """Synthetic 'hmda'-named data: approval depends on income only, never on race."""
    rng = np.random.default_rng(0)
    n = 1200
    income = rng.normal(0, 1, n)
    frame = pd.DataFrame({
        "income": income,
        "loan_type": rng.choice(["1", "2"], n),
        "race": rng.choice(["White", "Black or African American", "Asian"], n),
        "ethnicity": rng.choice(["Hispanic or Latino", "Not Hispanic or Latino"], n),
        "sex": rng.choice(["Male", "Female"], n),
        "age_group": rng.choice(["62+", "under 62"], n),
        "approved": (income + rng.normal(0, 0.5, n) > -0.8).astype(int),
    })
    frame.loc[::50, "race"] = pd.NA  # some applicants don't report race
    return Dataset("hmda", frame, ["income"], ["loan_type"],
                   ["race", "ethnicity", "sex", "age_group"])


def fit(name, dataset):
    return models.build_model(name, dataset).fit(dataset.frame)


@pytest.mark.parametrize("name", list(models.MODELS))
def test_interface(name, dataset):
    model = fit(name, dataset)
    proba = model.predict_proba(dataset.frame)
    pred = model.predict(dataset.frame)
    assert proba.shape == pred.shape == (len(dataset.frame),)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert set(np.unique(pred)) <= {0, 1}
    assert model.name == name and model.description


@pytest.mark.parametrize("name", ["vendor_a_logreg", "vendor_b_gbm"])
def test_honest_models_ignore_protected_columns(name, dataset):
    model = fit(name, dataset)
    shuffled = dataset.frame.copy()
    for col in dataset.protected:
        shuffled[col] = shuffled[col].sample(frac=1, random_state=1).to_numpy()
    np.testing.assert_array_equal(model.predict_proba(dataset.frame), model.predict_proba(shuffled))


def test_mitigated_model_uses_protected_columns(dataset):
    model = fit("vendor_a_mitigated", dataset)
    assert set(model._sensitive(dataset.frame)) <= model.cells | {model.SMALL}
    cells = model._cell(dataset.frame)
    assert cells.str.count(r"\|").eq(len(dataset.protected) - 1).all()
    assert cells.str.contains("NA").any()  # unreported race is its own category


def test_models_are_reproducible(dataset):
    a = fit("vendor_a_mitigated", dataset).predict(dataset.frame)
    b = fit("vendor_a_mitigated", dataset).predict(dataset.frame)
    np.testing.assert_array_equal(a, b)


def test_unknown_model_rejected(dataset):
    with pytest.raises(ValueError, match="unknown model"):
        models.build_model("vendor_z", dataset)
