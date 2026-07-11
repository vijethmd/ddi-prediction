"""Generate publication figures from the DEPLOYED model bundle.

Every figure is drawn from the same saved bundle (models/ddi_bundle.joblib) and
the same held-out test fold that scripts/train.py reports, so the figures and the
paper tables cannot drift apart. Numbers are real DrugBank 6.0 results, not
synthetic.
"""

import sys
sys.path.insert(0, "src")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.size"] = 10
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3
rcParams["figure.dpi"] = 150

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix

from ddi.enhance.metrics import reliability_curve, severe_recall

CLASSES = ["No Int", "Moderate", "Severe", "Synerg", "Antag"]
TEAL = "#0f6e56"
BLUE = "#185fa5"
RED = "#a32d2d"
AMBER = "#ba7517"
PURPLE = "#534ab7"
OUTDIR = "figs"
import os
os.makedirs(OUTDIR, exist_ok=True)

# ---- load the deployed bundle and reproduce train.py's test split ---------
print("Building figures from the deployed bundle on the real test fold...")
_blob = np.load("data/processed/features.npz", allow_pickle=True)
X, y, names = _blob["X"], _blob["y"].astype(int), [str(n) for n in _blob["names"]]
B = joblib.load("models/ddi_bundle.joblib")
clf, cal, conf, opt = (B["classifier"], B["calibrator"],
                       B["conformal"], B["severe_threshold"])
ki = B["kept_indices"]
print(f"Loaded {X.shape} real feature matrix; bundle kept {len(ki)} features")

# Exactly the split scripts/train.py evaluates on.
_, Xte, _, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                  random_state=42)
Xte_k = Xte[:, ki]
raw_test = clf.predict_proba(Xte_k)
cal_test = cal.transform(raw_test)
preds_thresh = opt.apply(cal_test)
preds_argmax = cal_test.argmax(axis=1)

print(f"  argmax severe recall: {severe_recall(yte, preds_argmax):.3f}")
print(f"  thresholded severe recall: {severe_recall(yte, preds_thresh):.3f}")

# ============================================================ FIG 1: recall-tau
print("Fig 1: recall/precision vs threshold")
sev = cal_test[:, 2]
taus = np.linspace(0.02, 0.7, 80)
recalls, precisions, f1s = [], [], []
for tau in taus:
    pred = cal_test.argmax(axis=1).copy()
    pred[sev >= tau] = 2
    tp = ((pred == 2) & (yte == 2)).sum()
    fp = ((pred == 2) & (yte != 2)).sum()
    fn = ((pred != 2) & (yte == 2)).sum()
    rec = tp / (tp + fn) if (tp + fn) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    recalls.append(rec)
    precisions.append(prec)
    f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0)

fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(taus, recalls, color=RED, lw=2, label="Severe recall")
ax.plot(taus, precisions, color=BLUE, lw=2, ls="--", label="Severe precision")
ax.plot(taus, f1s, color=TEAL, lw=1.5, ls=":", label="Severe F1")
ax.axhline(0.90, color="gray", lw=1, ls="-.", alpha=0.7)
ax.text(0.45, 0.915, "clinical floor 0.90", fontsize=8, color="gray")
ax.axvline(opt.tau_, color=AMBER, lw=1.5, alpha=0.8)
ax.text(opt.tau_ + 0.01, 0.15, f"selected\ntau={opt.tau_:.3f}",
        fontsize=8, color=AMBER)
ax.set_xlabel("Severe probability threshold  tau")
ax.set_ylabel("Score")
ax.set_title("Severe-class recall/precision vs. decision threshold")
ax.legend(loc="center right", fontsize=8, framealpha=0.9)
ax.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig_threshold.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================ FIG 2: reliability
print("Fig 2: reliability diagram")
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="perfect")
rc_raw = reliability_curve((yte == 2).astype(float), raw_test[:, 2], 10)
rc_cal = reliability_curve((yte == 2).astype(float), cal_test[:, 2], 10)
m1 = ~np.isnan(rc_raw["accuracy"])
m2 = ~np.isnan(rc_cal["accuracy"])
ax.plot(rc_raw["confidence"][m1], rc_raw["accuracy"][m1], "o-", color=RED,
        lw=1.8, ms=5, label="raw XGBoost")
ax.plot(rc_cal["confidence"][m2], rc_cal["accuracy"][m2], "s-", color=TEAL,
        lw=1.8, ms=5, label="isotonic-calibrated")
ax.set_xlabel("Mean predicted probability (Severe)")
ax.set_ylabel("Observed frequency")
ax.set_title("Reliability diagram — Severe class")
ax.legend(loc="upper left", fontsize=8)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig_reliability.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================ FIG 3: confusion
print("Fig 3: confusion matrices")
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
for ax, preds, title in [
    (axes[0], preds_argmax, "Argmax"),
    (axes[1], preds_thresh, "With Severe threshold"),
]:
    cm = confusion_matrix(yte, preds, labels=[0, 1, 2, 3, 4])
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cmn, cmap="BuGn", vmin=0, vmax=1)
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{title}\nSevere recall = {severe_recall(yte, preds):.3f}",
                 fontsize=10)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                    fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "#333")
    ax.grid(False)
fig.suptitle("Confusion matrices — the Severe row fills in after thresholding",
             fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig_confusion.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================ FIG 4: PR curves
print("Fig 4: PR curves")
fig, ax = plt.subplots(figsize=(4.8, 3.8))
colors = ["#888", AMBER, RED, BLUE, PURPLE]
for c in range(5):
    binary = (yte == c).astype(int)
    if binary.sum() < 2:
        continue
    prec, rec, _ = precision_recall_curve(binary, cal_test[:, c])
    from sklearn.metrics import auc
    ax.plot(rec, prec, color=colors[c], lw=1.8,
            label=f"{CLASSES[c]} (AP={auc(rec, prec):.2f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Per-class precision-recall curves")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig_pr.pdf", bbox_inches="tight")
plt.close(fig)

# ==================================================== FIG 5: per-class SHAP bars
# Per-class SHAP on the deployed Stage-2 classifier. The paper's claim is
# specifically that CYP features dominate the MODERATE (pharmacokinetic) class,
# so this figure shows that class's ranking, not a pooled one.
print("Fig 5: per-class SHAP feature importance (Moderate class)")
import shap
pruned_names = [names[i] for i in ki]
sub = Xte_k[np.random.default_rng(0).choice(len(Xte_k),
            min(3000, len(Xte_k)), replace=False)]
sv = np.asarray(shap.TreeExplainer(clf.stage2).shap_values(
    sub, check_additivity=False))
MODERATE_LOCAL = 0  # Stage-2 local index 0 -> Moderate
row = (np.abs(sv[:, :, MODERATE_LOCAL]).mean(axis=0) if sv.ndim == 3
       else np.abs(sv).mean(axis=0))
order = np.argsort(row)[::-1][:15]


def _pretty(n):
    return (n.replace("cyp_", "CYP:").replace("_", " ")
             .replace("physchem ", "").replace("ecfp4 ", "ECFP4 ")
             .replace("maccs ", "MACCS "))


fig, ax = plt.subplots(figsize=(5.8, 4.2))
labels = [_pretty(pruned_names[i]) for i in order][::-1]
vals = [row[i] for i in order][::-1]
cols = [RED if pruned_names[i].startswith("cyp_")
        else (BLUE if "tanimoto" in pruned_names[i] or "target" in pruned_names[i]
              else TEAL) for i in order][::-1]
ax.barh(range(len(vals)), vals, color=cols)
ax.set_yticks(range(len(vals)))
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel("Mean |SHAP value|  (Moderate class)")
ax.set_title("Top-15 features driving the Moderate class\n"
             "(red = CYP450, blue = Tanimoto/target overlap)")
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig_shap.pdf", bbox_inches="tight")
plt.close(fig)

# ============================================================ FIG 6: set sizes
print("Fig 6: conformal set-size distribution")
sets = conf.predict_set(cal_test)
sizes = np.array([len(s) for s in sets])
fig, ax = plt.subplots(figsize=(4.8, 3.4))
counts = np.bincount(sizes, minlength=6)[1:6]
bars = ax.bar(range(1, 6), counts, color=TEAL, alpha=0.85)
ax.set_xlabel("Prediction set size")
ax.set_ylabel("Number of test pairs")
ax.set_title("Conformal prediction set sizes\n"
             f"singleton rate = {(sizes==1).mean():.0%}")
for b, c in zip(bars, counts):
    if c > 0:
        ax.text(b.get_x() + b.get_width()/2, c, str(int(c)),
                ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/fig_setsize.pdf", bbox_inches="tight")
plt.close(fig)

print("\nAll figures written to", OUTDIR)
import os
for f in sorted(os.listdir(OUTDIR)):
    print("  ", f)
