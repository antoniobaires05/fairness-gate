import math

import numpy as np
import pytest
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

from ml.confidence import bootstrap_gaps, wilson_interval


# --- Wilson interval ---------------------------------------------------------


def test_wilson_matches_textbook_value():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)


def test_wilson_zero_successes_stays_inside_0_1():
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=1e-4)


def test_wilson_all_successes():
    lo, hi = wilson_interval(10, 10)
    assert hi == 1.0
    assert lo == pytest.approx(1 - 0.2775, abs=1e-4)


def test_wilson_empty_group_is_nan():
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


def test_wilson_narrows_with_more_data_and_widens_with_level():
    small = np.subtract(*wilson_interval(50, 100)[::-1])
    large = np.subtract(*wilson_interval(500, 1000)[::-1])
    wide = np.subtract(*wilson_interval(50, 100, level=0.99)[::-1])
    assert large < small < wide


# --- Bootstrap gap intervals ---------------------------------------------------


def two_groups(rate_a, rate_b, n=400, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=2 * n)
    y_pred = np.concatenate([rng.random(n) < rate_a, rng.random(n) < rate_b]).astype(int)
    groups = np.array(["a"] * n + ["b"] * n)
    return y_true, y_pred, groups


def test_interval_contains_point_estimate():
    y_true, y_pred, groups = two_groups(0.8, 0.5)
    ci = bootstrap_gaps(y_true, y_pred, groups, n_boot=500)
    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=groups)
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=groups)
    assert ci["demographic_parity_difference"][0] <= dpd <= ci["demographic_parity_difference"][1]
    assert ci["equalized_odds_difference"][0] <= eod <= ci["equalized_odds_difference"][1]


def test_large_gap_interval_excludes_zero_and_equal_groups_near_zero():
    biased = bootstrap_gaps(*two_groups(0.9, 0.3), n_boot=500)
    fair = bootstrap_gaps(*two_groups(0.6, 0.6), n_boot=500)
    assert biased["demographic_parity_difference"][0] > 0.4
    assert fair["demographic_parity_difference"][0] < 0.05


def test_bootstrap_is_reproducible_with_seed():
    data = two_groups(0.7, 0.6)
    assert bootstrap_gaps(*data, seed=1) == bootstrap_gaps(*data, seed=1)
    assert bootstrap_gaps(*data, seed=1) != bootstrap_gaps(*data, seed=2)


def test_identical_predictions_give_zero_width_interval():
    y_true = np.array([1, 0, 1, 0] * 10)
    y_pred = np.ones(40, dtype=int)
    groups = np.array(["a", "a", "b", "b"] * 10)
    ci = bootstrap_gaps(y_true, y_pred, groups, n_boot=200)
    assert ci["demographic_parity_difference"] == (0.0, 0.0)
    assert ci["equalized_odds_difference"] == (0.0, 0.0)


def test_group_without_negatives_does_not_crash():
    y_true = np.array([1, 1, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 1])
    groups = np.array(["a"] * 4 + ["b"] * 4)  # group a has no true denials -> FPR undefined
    lo, hi = bootstrap_gaps(y_true, y_pred, groups, n_boot=200)["equalized_odds_difference"]
    assert 0 <= lo <= hi <= 1
