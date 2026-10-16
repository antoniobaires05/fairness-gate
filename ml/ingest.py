"""The Dataset contract every cleaned dataset follows, and load_dataset(name).

The only dataset is HMDA (Delaware 2024); its cleaning lives in ml/ingest_hmda.py.
train.py, models.py and fairness_eval.py only see a Dataset, so adding or swapping a
dataset only touches the ingest layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Dataset:
    """The standard frame every dataset is cleaned into, plus how to use it."""

    name: str
    frame: pd.DataFrame
    numeric_features: list[str]
    categorical_features: list[str]
    protected: list[str]  # audit columns: never model features
    target: str = "approved"

    @property
    def features(self) -> list[str]:
        return self.numeric_features + self.categorical_features


DATASET_NAMES = ("hmda",)


def load_dataset(name: str) -> Dataset:
    """Load a dataset by name. Only 'hmda' exists."""
    if name == "hmda":
        from ml import ingest_hmda

        return ingest_hmda.load_dataset()
    raise ValueError(f"unknown dataset '{name}' (known: {', '.join(DATASET_NAMES)})")
