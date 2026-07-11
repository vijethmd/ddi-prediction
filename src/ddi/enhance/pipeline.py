from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.model_selection import train_test_split

from .pruning import ShapFeaturePruner, PruningResult
from .calibration import IsotonicCalibrator, calibration_report, CalibrationReport
from .conformal import MondrianConformalPredictor, ConformalResult
from .metrics import severe_recall, macro_f1


@dataclass
class PipelineConfig:
    n_classes: int = 5
    severe_label: int = 2
    coverage: float = 0.95
    protected_substrings: tuple = ("cyp", "tanimoto", "target_overlap")
    alpha_per_class: Optional[dict] = None
    conformal_score: str = "lac"
    calib_frac: float = 0.15
    conformal_frac: float = 0.15
    random_state: int = 42

    def __post_init__(self):
        if self.alpha_per_class is None:
            self.alpha_per_class = {0: 0.10, 1: 0.10, 2: 0.05, 3: 0.10, 4: 0.10}


@dataclass
class PredictionResult:
    label: int
    label_name: str
    calibrated_probs: np.ndarray
    prediction_set: List[int]
    prediction_set_names: List[str]
    confident: bool
    severe_in_set: bool

    def as_dict(self):
        return {
            "label": int(self.label),
            "label_name": self.label_name,
            "calibrated_probs": [round(float(p), 4) for p in self.calibrated_probs],
            "prediction_set": self.prediction_set,
            "prediction_set_names": self.prediction_set_names,
            "confident": self.confident,
            "severe_in_set": self.severe_in_set,
        }


class EnhancedDDIClassifier:
    """Wraps a base estimator factory and adds the three Phase 2 upgrades.

    Split discipline (no leakage):
        X_model    -> trains the base XGBoost
        X_calib    -> fits isotonic calibration (base never saw it)
        X_conf     -> calibrates conformal thresholds (calibrated probs)
        X_test     -> final held-out evaluation
    """

    CLASS_NAMES = {
        0: "No Interaction", 1: "Moderate", 2: "Severe",
        3: "Synergistic", 4: "Antagonistic",
    }

    def __init__(self, base_factory, config: PipelineConfig = None):
        self.base_factory = base_factory
        self.cfg = config or PipelineConfig()
        self.base_full = None
        self.base_pruned = None
        self.pruner = None
        self.pruning_result: Optional[PruningResult] = None
        self.calibrator: Optional[IsotonicCalibrator] = None
        self.conformal: Optional[MondrianConformalPredictor] = None
        self.calibration_report_: Optional[CalibrationReport] = None
        self.conformal_result_: Optional[ConformalResult] = None
        self.feature_names_ = None

    def _split(self, X, y):
        cfg = self.cfg
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=cfg.random_state)
        rel_calib = cfg.calib_frac / (1.0 - 0.20)
        rel_conf = cfg.conformal_frac / (1.0 - 0.20)
        X_model, X_hold, y_model, y_hold = train_test_split(
            X_tmp, y_tmp, test_size=rel_calib + rel_conf,
            stratify=y_tmp, random_state=cfg.random_state)
        conf_share = rel_conf / (rel_calib + rel_conf)
        X_calib, X_conf, y_calib, y_conf = train_test_split(
            X_hold, y_hold, test_size=conf_share,
            stratify=y_hold, random_state=cfg.random_state)
        return (X_model, y_model, X_calib, y_calib,
                X_conf, y_conf, X_test, y_test)

    def fit(self, X, y, feature_names, prune=True, verbose=True):
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        self.feature_names_ = list(feature_names)
        cfg = self.cfg

        (X_model, y_model, X_calib, y_calib,
         X_conf, y_conf, X_test, y_test) = self._split(X, y)

        if verbose:
            print(f"Splits  model={len(y_model)}  calib={len(y_calib)}  "
                  f"conf={len(y_conf)}  test={len(y_test)}")

        self.base_full = self.base_factory()
        self.base_full.fit(X_model, y_model)

        if prune:
            self.pruner = ShapFeaturePruner(
                coverage=cfg.coverage,
                protected_substrings=cfg.protected_substrings,
                random_state=cfg.random_state,
            )
            self.pruning_result = self.pruner.fit(
                self.base_full, X_model, self.feature_names_)
            if verbose:
                print(self.pruning_result.summary())
            kept = self.pruning_result.kept_indices
            self.base_pruned = self.base_factory()
            self.base_pruned.fit(X_model[:, kept], y_model)
            active = self.base_pruned
            sel = lambda Z: Z[:, kept]
        else:
            active = self.base_full
            sel = lambda Z: Z

        self.calibrator = IsotonicCalibrator(cfg.n_classes)
        raw_calib = active.predict_proba(sel(X_calib))
        self.calibrator.fit(raw_calib, y_calib)

        raw_test = active.predict_proba(sel(X_test))
        cal_test = self.calibrator.transform(raw_test)
        self.calibration_report_ = calibration_report(
            y_test, raw_test, cal_test, cfg.n_classes)
        if verbose:
            print(self.calibration_report_.summary(self.CLASS_NAMES))

        raw_conf = active.predict_proba(sel(X_conf))
        cal_conf = self.calibrator.transform(raw_conf)
        self.conformal = MondrianConformalPredictor(
            cfg.n_classes, alpha_per_class=cfg.alpha_per_class,
            score=cfg.conformal_score)
        self.conformal.calibrate(cal_conf, y_conf)

        self.conformal_result_ = self.conformal.evaluate(cal_test, y_test)
        if verbose:
            print(self.conformal_result_.summary(self.CLASS_NAMES))

        self._active = active
        self._sel = sel
        self._eval_cache = {
            "X_test": X_test, "y_test": y_test,
            "cal_test": cal_test, "raw_test": raw_test,
        }
        return self

    def evaluate_test(self):
        c = self._eval_cache
        preds_full = self.base_full.predict(c["X_test"])
        out = {
            "severe_recall_full": severe_recall(
                c["y_test"], preds_full, self.cfg.severe_label),
            "macro_f1_full": macro_f1(c["y_test"], preds_full),
        }
        if self.base_pruned is not None:
            kept = self.pruning_result.kept_indices
            preds_pruned = self.base_pruned.predict(c["X_test"][:, kept])
            out.update({
                "severe_recall_pruned": severe_recall(
                    c["y_test"], preds_pruned, self.cfg.severe_label),
                "macro_f1_pruned": macro_f1(c["y_test"], preds_pruned),
                "n_features_full": self.pruning_result.n_full,
                "n_features_pruned": self.pruning_result.n_kept,
                "reduction_pct": self.pruning_result.reduction_pct,
            })
        return out

    def predict(self, X):
        X = np.asarray(X)
        raw = self._active.predict_proba(self._sel(X))
        cal = self.calibrator.transform(raw)
        sets = self.conformal.predict_set(cal)
        results = []
        for i in range(len(X)):
            label = int(np.argmax(cal[i]))
            ps = sets[i]
            results.append(PredictionResult(
                label=label,
                label_name=self.CLASS_NAMES.get(label, str(label)),
                calibrated_probs=cal[i],
                prediction_set=ps,
                prediction_set_names=[self.CLASS_NAMES.get(c, str(c)) for c in ps],
                confident=(len(ps) == 1),
                severe_in_set=(self.cfg.severe_label in ps),
            ))
        return results
