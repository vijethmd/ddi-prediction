import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ddi.enhance import (
    IsotonicCalibrator, MondrianConformalPredictor,
    expected_calibration_error, brier_multiclass,
)
from ddi.enhance.metrics import severe_recall, macro_f1


def _fake_probs(n=500, n_classes=5, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1, (n, n_classes))
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(n_classes, p=probs[i]) for i in range(n)])
    return probs, y


def test_calibrator_rows_sum_to_one():
    probs, y = _fake_probs()
    cal = IsotonicCalibrator(5).fit(probs, y)
    out = cal.transform(probs)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-6)


def test_calibrator_reduces_or_holds_ece():
    probs, y = _fake_probs(seed=3)
    # Skew probs to induce miscalibration.
    skewed = probs ** 1.5
    skewed /= skewed.sum(axis=1, keepdims=True)
    cal = IsotonicCalibrator(5).fit(skewed, y)
    out = cal.transform(skewed)
    before = expected_calibration_error((y == 2).astype(float), skewed[:, 2])
    after = expected_calibration_error((y == 2).astype(float), out[:, 2])
    assert after <= before + 0.02


def test_conformal_coverage_holds():
    probs, y = _fake_probs(n=2000, seed=7)
    split = 1000
    cp = MondrianConformalPredictor(
        5, alpha_per_class={c: 0.1 for c in range(5)}, score="lac")
    cp.calibrate(probs[:split], y[:split])
    res = cp.evaluate(probs[split:], y[split:])
    # Overall coverage should be near or above 0.90 (allow small slack).
    assert res.overall_coverage >= 0.85


def test_conformal_sets_never_empty():
    probs, y = _fake_probs(n=800, seed=11)
    cp = MondrianConformalPredictor(5, score="aps")
    cp.calibrate(probs[:400], y[:400])
    sets = cp.predict_set(probs[400:])
    assert all(len(s) >= 1 for s in sets)


def test_conformal_quantile_finite_sample():
    # With alpha=0.1 and n=9 calib points, level = ceil(10*0.9)/9 = 1.0.
    q = MondrianConformalPredictor._conformal_quantile(
        np.linspace(0, 1, 9), 0.1)
    assert q == 1.0


def test_brier_bounds():
    probs, y = _fake_probs()
    b = brier_multiclass(y, probs)
    assert 0.0 <= b <= 2.0


def test_severe_recall_perfect():
    y = np.array([2, 2, 0, 1])
    assert severe_recall(y, y) == 1.0


def test_macro_f1_range():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    assert macro_f1(y_true, y_pred) == 1.0
