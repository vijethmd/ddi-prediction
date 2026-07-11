"""Evaluation metrics for the 5-class DDI classifier."""

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    f1_score,
    cohen_kappa_score,
    precision_recall_curve,
    auc,
)

from .. import CLASS_NAMES, SEVERE_LABEL


def full_report(y_true, y_pred, probs=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = sorted(CLASS_NAMES)
    names = [CLASS_NAMES[i] for i in labels]

    out = {
        "severe_recall": recall_score(
            y_true, y_pred, labels=[SEVERE_LABEL], average="macro",
            zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted",
                                zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": names,
        "report": classification_report(
            y_true, y_pred, labels=labels, target_names=names,
            zero_division=0, output_dict=True),
    }

    if probs is not None:
        probs = np.asarray(probs)
        pr_auc = {}
        for c in labels:
            binary = (y_true == c).astype(int)
            if binary.sum() == 0:
                pr_auc[CLASS_NAMES[c]] = float("nan")
                continue
            prec, rec, _ = precision_recall_curve(binary, probs[:, c])
            pr_auc[CLASS_NAMES[c]] = float(auc(rec, prec))
        out["pr_auc_per_class"] = pr_auc
    return out


def print_report(report):
    print(f"Severe recall : {report['severe_recall']:.3f}")
    print(f"Macro F1      : {report['macro_f1']:.3f}")
    print(f"Weighted F1   : {report['weighted_f1']:.3f}")
    print(f"Cohen's kappa : {report['cohen_kappa']:.3f}")
    if "pr_auc_per_class" in report:
        print("PR-AUC per class:")
        for k, v in report["pr_auc_per_class"].items():
            print(f"  {k:<16} {v:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    header = "            " + " ".join(f"{n[:8]:>8}" for n in report["labels"])
    print(header)
    for name, row in zip(report["labels"], report["confusion_matrix"]):
        print(f"  {name[:10]:<10} " + " ".join(f"{v:>8}" for v in row))
