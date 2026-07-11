import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ddi.enhance import SevereThresholdOptimizer
from ddi.enhance.metrics import severe_recall


def _make_probs(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1, (n, 5))
    # Make Severe (class 2) genuinely present but under-predicted by argmax.
    logits[:, 2] += rng.normal(-0.5, 1.0, n)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(5, p=probs[i]) for i in range(n)])
    return probs, y


def test_threshold_raises_severe_recall():
    probs, y = _make_probs(seed=1)
    split = 500
    opt = SevereThresholdOptimizer(target_recall=0.90)
    opt.fit(probs[:split], y[:split])
    tr = opt.tradeoff(probs[split:], y[split:])
    # Thresholded recall should be >= plain argmax recall.
    assert tr["severe_recall_thresholded"] >= tr["severe_recall_plain"]


def test_threshold_apply_shape():
    probs, y = _make_probs()
    opt = SevereThresholdOptimizer(target_recall=0.85).fit(probs, y)
    preds = opt.apply(probs)
    assert preds.shape == (len(probs),)
    assert set(np.unique(preds)).issubset({0, 1, 2, 3, 4})


def test_threshold_promotes_high_severe_prob():
    opt = SevereThresholdOptimizer(target_recall=0.5)
    # Two samples: one clearly non-severe, one with high severe prob.
    probs = np.array([[0.7, 0.1, 0.1, 0.05, 0.05],
                      [0.4, 0.1, 0.45, 0.03, 0.02]])
    y = np.array([0, 2])
    opt.fit(probs, y)
    preds = opt.apply(probs)
    # The high-severe-prob sample must be labelled Severe.
    assert preds[1] == 2


def test_unreachable_target_uses_floor():
    probs, y = _make_probs()
    # Target impossible (1.0) -> optimizer falls back to floor, still runs.
    opt = SevereThresholdOptimizer(target_recall=1.0).fit(probs, y)
    assert opt.tau_ is not None
    assert opt.result_.summary()


def test_apply_before_fit_raises():
    opt = SevereThresholdOptimizer()
    try:
        opt.apply(np.zeros((3, 5)))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
