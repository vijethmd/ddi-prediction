import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PruningResult:
    kept_indices: np.ndarray
    kept_names: list
    importances: np.ndarray
    n_full: int
    n_kept: int
    coverage: float
    protected_kept: list = field(default_factory=list)
    explain_seconds: float = 0.0

    @property
    def reduction_pct(self):
        return 100.0 * (1.0 - self.n_kept / self.n_full)

    def summary(self):
        return (
            f"Features {self.n_full} -> {self.n_kept} "
            f"({self.reduction_pct:.1f}% reduction, {self.coverage:.0%} coverage). "
            f"Protected retained: {len(self.protected_kept)}. "
            f"SHAP time {self.explain_seconds:.2f}s."
        )


class ShapFeaturePruner:
    """Rank features by mean |SHAP| and keep the smallest set covering a target
    fraction of total importance. Multi-class aware. Memory-safe via batching.
    Features whose name matches a protected substring are always retained."""

    def __init__(
        self,
        coverage=0.95,
        protected_substrings=("cyp", "tanimoto", "target_overlap"),
        batch_size=2000,
        min_features=50,
        random_state=42,
    ):
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage must be in (0, 1]")
        self.coverage = coverage
        self.protected_substrings = tuple(s.lower() for s in protected_substrings)
        self.batch_size = batch_size
        self.min_features = min_features
        self.random_state = random_state

    def _mean_abs_shap(self, model, X):
        import shap

        explainer = shap.TreeExplainer(model)
        n, d = X.shape
        acc = np.zeros(d, dtype=np.float64)
        seen = 0
        for start in range(0, n, self.batch_size):
            chunk = X[start : start + self.batch_size]
            sv = explainer.shap_values(chunk, check_additivity=False)
            acc += self._reduce_shap(sv, d)
            seen += chunk.shape[0]
        return acc / max(seen, 1)

    @staticmethod
    def _reduce_shap(sv, d):
        # Handle every shap return shape: list per class, 3D array, or 2D.
        if isinstance(sv, list):
            out = np.zeros(d, dtype=np.float64)
            for class_sv in sv:
                out += np.abs(class_sv).sum(axis=0)
            return out
        sv = np.asarray(sv)
        if sv.ndim == 3:  # (n, d, classes) or (classes, n, d)
            if sv.shape[1] == d:
                return np.abs(sv).sum(axis=(0, 2))
            return np.abs(sv).sum(axis=(0, 1))
        return np.abs(sv).sum(axis=0)

    def _protected_mask(self, feature_names):
        mask = np.zeros(len(feature_names), dtype=bool)
        for i, name in enumerate(feature_names):
            low = str(name).lower()
            if any(sub in low for sub in self.protected_substrings):
                mask[i] = True
        return mask

    def fit(self, model, X_train, feature_names):
        X_train = np.asarray(X_train)
        d = X_train.shape[1]
        if len(feature_names) != d:
            raise ValueError("feature_names length must match X_train columns")

        t0 = time.perf_counter()
        importances = self._mean_abs_shap(model, X_train)
        elapsed = time.perf_counter() - t0

        total = importances.sum()
        if total <= 0:
            order = np.arange(d)
            kept = order[: max(self.min_features, 1)]
            return PruningResult(
                kept_indices=kept,
                kept_names=[feature_names[i] for i in kept],
                importances=importances,
                n_full=d, n_kept=len(kept), coverage=0.0,
                protected_kept=[], explain_seconds=elapsed,
            )

        order = np.argsort(importances)[::-1]
        cum = np.cumsum(importances[order]) / total
        n_cov = int(np.searchsorted(cum, self.coverage) + 1)
        n_cov = max(n_cov, self.min_features)
        n_cov = min(n_cov, d)
        selected = set(order[:n_cov].tolist())

        protected_mask = self._protected_mask(feature_names)
        protected_idx = np.where(protected_mask)[0]
        for idx in protected_idx:
            selected.add(int(idx))

        kept = np.array(sorted(selected))
        achieved = importances[kept].sum() / total
        protected_kept = [feature_names[i] for i in kept if protected_mask[i]]

        return PruningResult(
            kept_indices=kept,
            kept_names=[feature_names[i] for i in kept],
            importances=importances,
            n_full=d, n_kept=len(kept), coverage=float(achieved),
            protected_kept=protected_kept, explain_seconds=elapsed,
        )

    @staticmethod
    def transform(X, result: PruningResult):
        return np.asarray(X)[:, result.kept_indices]

    def top_k_table(self, result: PruningResult, k=20):
        order = np.argsort(result.importances)[::-1][:k]
        rows = []
        total = result.importances.sum()
        for rank, idx in enumerate(order, 1):
            rows.append({
                "rank": rank,
                "feature": result.kept_names[
                    list(result.kept_indices).index(idx)
                ] if idx in result.kept_indices else f"feat_{idx}",
                "mean_abs_shap": float(result.importances[idx]),
                "pct_of_total": 100.0 * result.importances[idx] / total,
            })
        return rows
