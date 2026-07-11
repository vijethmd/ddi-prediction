"""Two-stage hierarchical DDI classifier.

Stage 1 (safety gate): binary XGBoost, decision threshold 0.30 (not 0.50) to
maximise recall of any-interaction. Pairs scoring >= 0.30 pass to Stage 2.

Stage 2 (type classifier): 4-class XGBoost over {Moderate, Severe, Synergistic,
Antagonistic} with class weights emphasising Severe (5.0). ADASYN oversampling is
applied to the Stage 2 training data before fitting.

The design separates the majority No-Interaction class from the minority
interaction types so each stage optimises its own objective without the majority
class swamping the loss.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .. import SEVERE_LABEL, CLASS_NAMES

STAGE2_CLASSES = [1, 2, 3, 4]  # Moderate, Severe, Synergistic, Antagonistic
STAGE2_WEIGHTS = {1: 1.5, 2: 5.0, 3: 3.0, 4: 3.0}


@dataclass
class TwoStageConfig:
    gate_threshold: float = 0.30
    use_adasyn: bool = True
    stage1_params: dict = field(default_factory=lambda: dict(
        n_estimators=400, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.8, tree_method="hist",
        n_jobs=-1, eval_metric="logloss", random_state=42))
    stage2_params: dict = field(default_factory=lambda: dict(
        n_estimators=500, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.8, tree_method="hist",
        n_jobs=-1, eval_metric="mlogloss", random_state=42))


class TwoStageDDIClassifier:
    def __init__(self, config: TwoStageConfig = None):
        self.cfg = config or TwoStageConfig()
        self.stage1 = None
        self.stage2 = None
        self._stage2_label_map = {c: i for i, c in enumerate(STAGE2_CLASSES)}
        self._stage2_inverse = {i: c for c, i in self._stage2_label_map.items()}

    def _import_xgb(self):
        import xgboost as xgb
        return xgb

    def _maybe_adasyn(self, X, y):
        if not self.cfg.use_adasyn:
            return X, y
        try:
            from imblearn.over_sampling import ADASYN
            counts = np.bincount(y)
            minority = counts[counts > 0].min()
            if minority < 6:
                return X, y  # too few for k-neighbours
            n_neighbors = min(5, minority - 1)
            ada = ADASYN(random_state=42, n_neighbors=n_neighbors)
            return ada.fit_resample(X, y)
        except Exception:
            return X, y

    def fit(self, X, y):
        xgb = self._import_xgb()
        X = np.asarray(X)
        y = np.asarray(y).astype(int)

        # Stage 1: any interaction vs none.
        y_binary = (y != 0).astype(int)
        pos = max((y_binary == 1).sum(), 1)
        neg = max((y_binary == 0).sum(), 1)
        self.stage1 = xgb.XGBClassifier(
            scale_pos_weight=neg / pos, **self.cfg.stage1_params)
        self.stage1.fit(X, y_binary)

        # Stage 2: type among interacting pairs.
        mask = y != 0
        X2, y2 = X[mask], y[mask]
        y2_local = np.array([self._stage2_label_map[c] for c in y2])
        X2r, y2r = self._maybe_adasyn(X2, y2_local)

        sample_weight = np.array(
            [STAGE2_WEIGHTS[self._stage2_inverse[c]] for c in y2r])
        n_stage2_classes = len(np.unique(y2r))
        stage2_kwargs = dict(self.cfg.stage2_params)
        if n_stage2_classes > 2:
            stage2_kwargs["objective"] = "multi:softprob"
        self.stage2 = xgb.XGBClassifier(**stage2_kwargs)
        self.stage2.fit(X2r, y2r, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        """Full 5-class probs; bulletproof to Stage-2 column/class mismatch."""
        X = np.asarray(X)
        gate = np.asarray(self.stage1.predict_proba(X)[:, 1]).ravel()
        n = X.shape[0]
        probs = np.zeros((n, 5))
        probs[:, 0] = 1.0 - gate
        s2 = np.asarray(self.stage2.predict_proba(X))
        if s2.ndim == 1:
            s2 = s2.reshape(n, -1)
        trained_local = [int(c) for c in np.asarray(self.stage2.classes_).ravel()]
        if len(trained_local) == 1:
            cls = self._stage2_inverse[trained_local[0]]
            col = 0 if s2.shape[1] == 1 else int(np.argmax(s2.sum(axis=0)))
            probs[:, cls] = gate * s2[:, col]
        else:
            k = min(len(trained_local), s2.shape[1])
            for col in range(k):
                cls = self._stage2_inverse[trained_local[col]]
                probs[:, cls] = gate * s2[:, col]
        row = probs.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        return probs / row

    def predict(self, X):
        X = np.asarray(X)
        gate = self.stage1.predict_proba(X)[:, 1]
        passed = gate >= self.cfg.gate_threshold
        out = np.zeros(X.shape[0], dtype=int)
        if passed.any():
            s2 = self.stage2.predict(X[passed])
            out[passed] = [self._stage2_inverse[i] for i in s2]
        return out

    def save(self, path):
        import joblib
        joblib.dump({
            "cfg": self.cfg,
            "stage1": self.stage1,
            "stage2": self.stage2,
            "map": self._stage2_label_map,
        }, path)

    @classmethod
    def load(cls, path):
        import joblib
        blob = joblib.load(path)
        obj = cls(blob["cfg"])
        obj.stage1 = blob["stage1"]
        obj.stage2 = blob["stage2"]
        obj._stage2_label_map = blob["map"]
        obj._stage2_inverse = {i: c for c, i in blob["map"].items()}
        return obj
