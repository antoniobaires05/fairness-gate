"""Common interface for credit models, plus stand-in "vendor" models.

Every model sees the full cleaned frame (features AND protected columns), the way
a third-party vendor's black box would receive an application file. The honest
models only read dataset.features.

Models:
    vendor_a_logreg     Vendor A: logistic-regression scorecard (the baseline)
    vendor_b_gbm        Vendor B: gradient-boosted trees
    vendor_a_mitigated  Vendor A post-processed by Fairlearn's ThresholdOptimizer
                        (equalized odds), i.e. a fairness-remediated challenger
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ml.ingest import Dataset

SEED = 42

# The mitigated challenger equalizes error rates across the intersection of all
# protected columns (equalizing one attribute alone just moved the gap to another).
# Intersections with fewer training rows than this are pooled into one group, so
# thresholds are never fitted to a handful of applicants.
MITIGATION_MIN_CELL = {"hmda": 100}


class CreditModel(ABC):
    """Fit on a cleaned frame; predict approval (1) or denial (0)."""

    name: str
    description: str

    def __init__(self, dataset: Dataset, seed: int = SEED):
        self.dataset = dataset
        self.seed = seed

    @abstractmethod
    def fit(self, frame: pd.DataFrame) -> "CreditModel": ...

    @abstractmethod
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Probability of approval for each row, shape (n,)."""

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(frame) >= 0.5).astype(int)


def _as_text(X) -> pd.DataFrame:
    """Categorical codes (int or str) -> text, with missing values as their own category."""
    return pd.DataFrame(X).astype("string").fillna("missing").astype(str)


def _preprocessor(dataset: Dataset) -> ColumnTransformer:
    numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    categorical = make_pipeline(
        FunctionTransformer(_as_text),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    return ColumnTransformer([
        ("num", numeric, dataset.numeric_features),
        ("cat", categorical, dataset.categorical_features),
    ])


class _SklearnModel(CreditModel):
    """A preprocessing + classifier pipeline that reads only dataset.features."""

    def _classifier(self):
        raise NotImplementedError

    def fit(self, frame):
        self.pipeline = Pipeline([("prep", _preprocessor(self.dataset)), ("clf", self._classifier())])
        self.pipeline.fit(frame[self.dataset.features], frame[self.dataset.target])
        return self

    def predict_proba(self, frame):
        return self.pipeline.predict_proba(frame[self.dataset.features])[:, 1]


class VendorALogReg(_SklearnModel):
    name = "vendor_a_logreg"
    description = "Vendor A: logistic-regression scorecard"

    def _classifier(self):
        return LogisticRegression(max_iter=2000, random_state=self.seed)


class VendorBGBM(_SklearnModel):
    name = "vendor_b_gbm"
    description = "Vendor B: gradient-boosted trees"

    def _classifier(self):
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, random_state=self.seed
        )


class VendorAMitigated(CreditModel):
    """Vendor A + group-specific thresholds chosen to equalize TPR and FPR."""

    name = "vendor_a_mitigated"
    description = "Vendor A + Fairlearn ThresholdOptimizer (equalized odds, intersectional)"
    SMALL = "pooled small groups"

    def _cell(self, frame) -> pd.Series:
        # One label per intersection, e.g. "White|Not Hispanic or Latino|Male|under 62";
        # unreported values count as their own category ("NA").
        cols = self.dataset.protected
        return frame[cols].astype("string").fillna("NA").agg("|".join, axis=1)

    def _sensitive(self, frame):
        cell = self._cell(frame)
        return cell.where(cell.isin(self.cells), self.SMALL)

    def fit(self, frame):
        counts = self._cell(frame).value_counts()
        self.cells = set(counts[counts >= MITIGATION_MIN_CELL[self.dataset.name]].index)
        base = VendorALogReg(self.dataset, self.seed).fit(frame)
        self.optimizer = ThresholdOptimizer(
            estimator=base.pipeline,
            constraints="equalized_odds",
            objective="accuracy_score",
            prefit=True,
            predict_method="predict_proba",
        )
        self.optimizer.fit(
            frame[self.dataset.features], frame[self.dataset.target],
            sensitive_features=self._sensitive(frame),
        )
        return self

    def predict_proba(self, frame):
        # Probability that the randomized threshold approves this applicant.
        pmf = self.optimizer._pmf_predict(
            frame[self.dataset.features], sensitive_features=self._sensitive(frame)
        )
        return pmf[:, 1]

    def predict(self, frame):
        return self.optimizer.predict(
            frame[self.dataset.features], sensitive_features=self._sensitive(frame),
            random_state=self.seed,
        ).astype(int)


MODELS: dict[str, type[CreditModel]] = {
    cls.name: cls for cls in (VendorALogReg, VendorBGBM, VendorAMitigated)
}


def build_model(name: str, dataset: Dataset, seed: int = SEED) -> CreditModel:
    try:
        return MODELS[name](dataset, seed)
    except KeyError:
        raise ValueError(f"unknown model '{name}' (known: {', '.join(MODELS)})") from None
