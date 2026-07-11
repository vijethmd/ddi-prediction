import numpy as np
from sklearn.metrics import recall_score, f1_score


def severe_recall(y_true, y_pred, severe_label=2):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if (y_true == severe_label).sum() == 0:
        return float("nan")
    return recall_score(
        y_true, y_pred, labels=[severe_label], average="macro", zero_division=0
    )


def macro_f1(y_true, y_pred):
    return f1_score(np.asarray(y_true), np.asarray(y_pred),
                    average="macro", zero_division=0)


def expected_calibration_error(y_true_binary, y_prob, n_bins=10):
    y_true_binary = np.asarray(y_true_binary).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_prob)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob > lo) & (y_prob <= hi)
        count = mask.sum()
        if count == 0:
            continue
        acc = y_true_binary[mask].mean()
        conf = y_prob[mask].mean()
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def brier_multiclass(y_true, probs, n_classes=None):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    if n_classes is None:
        n_classes = probs.shape[1]
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def reliability_curve(y_true_binary, y_prob, n_bins=10):
    y_true_binary = np.asarray(y_true_binary).astype(float)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob > lo) & (y_prob <= hi)
        c = mask.sum()
        centers.append((lo + hi) / 2.0)
        counts.append(int(c))
        if c == 0:
            accs.append(np.nan)
            confs.append(np.nan)
        else:
            accs.append(float(y_true_binary[mask].mean()))
            confs.append(float(y_prob[mask].mean()))
    return {
        "bin_centers": np.array(centers),
        "accuracy": np.array(accs),
        "confidence": np.array(confs),
        "counts": np.array(counts),
    }
