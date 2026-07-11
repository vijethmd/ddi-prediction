from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class ConformalResult:
    coverage: Dict[int, float]
    target_coverage: Dict[int, float]
    mean_set_size: Dict[int, float]
    overall_coverage: float
    overall_mean_set_size: float
    singleton_rate: float
    empty_rate_before_fallback: float = 0.0

    def summary(self, class_names=None):
        lines = ["Mondrian conformal coverage (per class):"]
        for c in sorted(self.coverage):
            name = class_names.get(c, f"class {c}") if class_names else f"class {c}"
            tgt = self.target_coverage.get(c, float("nan"))
            lines.append(
                f"  {name:<16} target {tgt:.2f}  empirical {self.coverage[c]:.3f}"
                f"  mean|set| {self.mean_set_size[c]:.2f}"
            )
        lines.append(
            f"Overall coverage {self.overall_coverage:.3f}, "
            f"singleton rate {self.singleton_rate:.2f}"
        )
        return "\n".join(lines)


class MondrianConformalPredictor:
    """Class-conditional (Mondrian) split conformal prediction.

    Nonconformity score for a candidate label y at input x:
        score = 1 - p(y | x)              (default, 'lac')
    or APS (adaptive prediction sets):
        score = cumulative prob mass of classes at least as likely as y.

    Per-class threshold q_c is the finite-sample conformal quantile computed
    only from calibration examples whose TRUE label is c. The guarantee is then
    class-conditional: P(Y in C(X) | Y=c) >= 1 - alpha_c.
    """

    def __init__(self, n_classes, alpha_per_class=None, default_alpha=0.10,
                 score="lac"):
        if score not in ("lac", "aps"):
            raise ValueError("score must be 'lac' or 'aps'")
        self.n_classes = n_classes
        self.default_alpha = default_alpha
        self.alpha = {c: default_alpha for c in range(n_classes)}
        if alpha_per_class:
            self.alpha.update(alpha_per_class)
        self.score = score
        self.thresholds: Dict[int, float] = {}
        self._fitted = False

    def _nonconformity(self, probs, label):
        probs = np.asarray(probs, dtype=float)
        if self.score == "lac":
            return 1.0 - probs[:, label]
        # APS: sum of probabilities of classes ranked above the true label,
        # plus the true label's own probability.
        scores = np.empty(probs.shape[0])
        order = np.argsort(-probs, axis=1)
        for i in range(probs.shape[0]):
            ranking = order[i]
            cumulative = 0.0
            for cls in ranking:
                cumulative += probs[i, cls]
                if cls == label:
                    break
            scores[i] = cumulative
        return scores

    def _aps_set_scores(self, prob_row):
        order = np.argsort(-prob_row)
        cumulative = 0.0
        out = {}
        for cls in order:
            cumulative += prob_row[cls]
            out[int(cls)] = cumulative
        return out

    @staticmethod
    def _conformal_quantile(scores, alpha):
        n = len(scores)
        if n == 0:
            return np.inf
        level = np.ceil((n + 1) * (1.0 - alpha)) / n
        level = min(level, 1.0)
        return float(np.quantile(scores, level, method="higher"))

    def calibrate(self, calib_probs, calib_labels):
        calib_probs = np.asarray(calib_probs, dtype=float)
        calib_labels = np.asarray(calib_labels).astype(int)
        for c in range(self.n_classes):
            mask = calib_labels == c
            if mask.sum() == 0:
                self.thresholds[c] = np.inf
                continue
            if self.score == "lac":
                scores = 1.0 - calib_probs[mask, c]
            else:
                scores = self._nonconformity_block(calib_probs[mask], c)
            self.thresholds[c] = self._conformal_quantile(scores, self.alpha[c])
        self._fitted = True
        return self

    def _nonconformity_block(self, probs_block, label):
        out = np.empty(probs_block.shape[0])
        for i in range(probs_block.shape[0]):
            s = self._aps_set_scores(probs_block[i])
            out[i] = s[label]
        return out

    def predict_set(self, probs):
        if not self._fitted:
            raise RuntimeError("call calibrate() before predict_set()")
        probs = np.asarray(probs, dtype=float)
        sets: List[List[int]] = []
        empties = 0
        for row in probs:
            if self.score == "lac":
                members = [c for c in range(self.n_classes)
                           if (1.0 - row[c]) <= self.thresholds[c]]
            else:
                set_scores = self._aps_set_scores(row)
                members = [c for c in range(self.n_classes)
                           if set_scores[c] <= self.thresholds[c]]
            if not members:
                empties += 1
                members = [int(np.argmax(row))]
            sets.append(sorted(members))
        self._last_empty_count = empties
        return sets

    def evaluate(self, test_probs, test_labels):
        test_probs = np.asarray(test_probs, dtype=float)
        test_labels = np.asarray(test_labels).astype(int)
        sets = self.predict_set(test_probs)
        empties = getattr(self, "_last_empty_count", 0)

        coverage, mean_size = {}, {}
        for c in range(self.n_classes):
            mask = test_labels == c
            if mask.sum() == 0:
                continue
            idxs = np.where(mask)[0]
            hits = sum(c in sets[i] for i in idxs)
            coverage[c] = hits / mask.sum()
            mean_size[c] = float(np.mean([len(sets[i]) for i in idxs]))

        all_sizes = [len(s) for s in sets]
        overall_cov = np.mean([test_labels[i] in sets[i] for i in range(len(sets))])
        singleton = np.mean([len(s) == 1 for s in sets])

        return ConformalResult(
            coverage=coverage,
            target_coverage={c: 1.0 - self.alpha[c] for c in coverage},
            mean_set_size=mean_size,
            overall_coverage=float(overall_cov),
            overall_mean_set_size=float(np.mean(all_sizes)),
            singleton_rate=float(singleton),
            empty_rate_before_fallback=empties / max(len(sets), 1),
        )
