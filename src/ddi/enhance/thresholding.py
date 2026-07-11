"""Severe-recall-targeted decision thresholding.

Standard argmax classification optimises overall accuracy, which underweights the
minority Severe class. For a clinical DDI tool the operating point that matters is
Severe recall: catching dangerous interactions is worth accepting more false
positives. This module selects a threshold tau on P(Severe) that achieves a
target Severe recall on a validation set, then applies it at inference by
promoting any pair with P(Severe) >= tau to the Severe class.

The threshold is chosen on validation data and applied to unseen data, so the
reported test-set gain is honest (no leakage). The macro-F1 cost of the
promotion is reported alongside so the tradeoff is explicit.
"""

from dataclasses import dataclass

import numpy as np

from .. import SEVERE_LABEL
from .metrics import severe_recall, macro_f1


@dataclass
class ThresholdResult:
    tau: float
    achieved_val_recall: float
    target_recall: float
    grid_size: int

    def summary(self):
        return (f"Severe threshold tau={self.tau:.3f} "
                f"(val recall {self.achieved_val_recall:.3f} "
                f"vs target {self.target_recall:.2f})")


class SevereThresholdOptimizer:
    """Pick the smallest P(Severe) threshold on validation data that reaches a
    target Severe recall. Applying a smaller threshold promotes more pairs to
    Severe, raising recall at the cost of precision."""

    def __init__(self, target_recall=0.90, severe_label=SEVERE_LABEL,
                 grid=None, floor=0.02):
        self.target_recall = target_recall
        self.severe_label = severe_label
        self.grid = grid if grid is not None else np.linspace(0.02, 0.7, 120)
        self.floor = floor
        self.tau_ = None
        self.result_ = None

    def fit(self, val_probs, val_labels):
        val_probs = np.asarray(val_probs, dtype=float)
        val_labels = np.asarray(val_labels).astype(int)
        sev = val_probs[:, self.severe_label]

        chosen = None
        achieved = 0.0
        # Ascending grid: the smallest tau meeting target keeps precision highest.
        for tau in self.grid:
            pred = val_probs.argmax(axis=1)
            pred[sev >= tau] = self.severe_label
            r = severe_recall(val_labels, pred, self.severe_label)
            if r >= self.target_recall:
                chosen, achieved = tau, r
        if chosen is None:
            # Cannot reach target anywhere; use the floor (max recall available).
            chosen = self.floor
            pred = val_probs.argmax(axis=1)
            pred[sev >= chosen] = self.severe_label
            achieved = severe_recall(val_labels, pred, self.severe_label)

        self.tau_ = float(chosen)
        self.result_ = ThresholdResult(
            tau=self.tau_, achieved_val_recall=float(achieved),
            target_recall=self.target_recall, grid_size=len(self.grid))
        return self

    def apply(self, probs):
        if self.tau_ is None:
            raise RuntimeError("call fit() before apply()")
        probs = np.asarray(probs, dtype=float)
        pred = probs.argmax(axis=1)
        pred[probs[:, self.severe_label] >= self.tau_] = self.severe_label
        return pred

    def tradeoff(self, test_probs, test_labels):
        """Report Severe recall and macro F1 with and without the threshold."""
        test_labels = np.asarray(test_labels).astype(int)
        plain = np.asarray(test_probs).argmax(axis=1)
        promoted = self.apply(test_probs)
        return {
            "severe_recall_plain": severe_recall(test_labels, plain,
                                                 self.severe_label),
            "severe_recall_thresholded": severe_recall(test_labels, promoted,
                                                       self.severe_label),
            "macro_f1_plain": macro_f1(test_labels, plain),
            "macro_f1_thresholded": macro_f1(test_labels, promoted),
            "tau": self.tau_,
        }
