from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .metrics import (
    expected_calibration_error,
    brier_multiclass,
    reliability_curve,
)


@dataclass
class CalibrationReport:
    ece_before: Dict[int, float]
    ece_after: Dict[int, float]
    brier_before: float
    brier_after: float
    reliability_before: Dict[int, dict]
    reliability_after: Dict[int, dict]

    def summary(self, class_names=None):
        lines = ["Calibration report (per-class ECE):"]
        for c in sorted(self.ece_before):
            name = class_names.get(c, f"class {c}") if class_names else f"class {c}"
            lines.append(
                f"  {name:<16} ECE {self.ece_before[c]:.3f} -> {self.ece_after[c]:.3f}"
            )
        lines.append(
            f"Multiclass Brier {self.brier_before:.4f} -> {self.brier_after:.4f}"
        )
        return "\n".join(lines)


class IsotonicCalibrator:
    """One isotonic regressor per class fit on raw class probabilities, then
    renormalised so calibrated rows sum to 1. Fit strictly on a held-out
    calibration set the base model never trained on."""

    def __init__(self, n_classes, clip_eps=1e-6):
        self.n_classes = n_classes
        self.clip_eps = clip_eps
        self.models: Dict[int, IsotonicRegression] = {}
        self._fitted = False

    def fit(self, calib_probs, calib_labels):
        calib_probs = np.asarray(calib_probs, dtype=float)
        calib_labels = np.asarray(calib_labels).astype(int)
        if calib_probs.shape[1] != self.n_classes:
            raise ValueError("calib_probs column count must equal n_classes")
        for c in range(self.n_classes):
            target = (calib_labels == c).astype(float)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            if target.sum() == 0:
                # degenerate: class absent in calibration fold -> identity-ish
                iso.fit(calib_probs[:, c], np.zeros_like(target))
            else:
                iso.fit(calib_probs[:, c], target)
            self.models[c] = iso
        self._fitted = True
        return self

    def transform(self, probs):
        if not self._fitted:
            raise RuntimeError("call fit() before transform()")
        probs = np.asarray(probs, dtype=float)
        out = np.zeros_like(probs)
        for c in range(self.n_classes):
            out[:, c] = self.models[c].predict(probs[:, c])
        out = np.clip(out, self.clip_eps, None)
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return out / row_sums

    def fit_transform(self, calib_probs, calib_labels):
        return self.fit(calib_probs, calib_labels).transform(calib_probs)


def calibration_report(
    y_true, raw_probs, calibrated_probs, n_classes, n_bins=10
) -> CalibrationReport:
    y_true = np.asarray(y_true).astype(int)
    raw_probs = np.asarray(raw_probs, dtype=float)
    calibrated_probs = np.asarray(calibrated_probs, dtype=float)

    ece_before, ece_after = {}, {}
    rel_before, rel_after = {}, {}
    for c in range(n_classes):
        bin_y = (y_true == c).astype(float)
        ece_before[c] = expected_calibration_error(bin_y, raw_probs[:, c], n_bins)
        ece_after[c] = expected_calibration_error(bin_y, calibrated_probs[:, c], n_bins)
        rel_before[c] = reliability_curve(bin_y, raw_probs[:, c], n_bins)
        rel_after[c] = reliability_curve(bin_y, calibrated_probs[:, c], n_bins)

    return CalibrationReport(
        ece_before=ece_before,
        ece_after=ece_after,
        brier_before=brier_multiclass(y_true, raw_probs, n_classes),
        brier_after=brier_multiclass(y_true, calibrated_probs, n_classes),
        reliability_before=rel_before,
        reliability_after=rel_after,
    )
