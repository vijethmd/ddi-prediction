"""Drug-disjoint (cold-start) ablation: what actually generalises to new drugs?

The random pair-level split used for the headline metrics lets a drug appear in
both the training and test folds. A model can then memorise "anything containing
warfarin is Severe" from its fingerprint bits and score well without learning any
interaction chemistry. That is precisely the cold-start failure this project
claims to solve, and a pair-level split cannot measure it.

Here the split is on DRUG IDENTITY: a set of drugs is held out entirely, and the
test fold contains only pairs where BOTH drugs are unseen. Under that regime we
compare feature groups:

    full            all 2,423 features
    no_cyp          CYP block zeroed (25 cols)
    no_fingerprint  ECFP4 + MACCS zeroed (2,382 cols)
    cyp_only        CYP block + tanimoto + target overlap (27 cols)

If CYP features are what carry cold-start performance, `no_cyp` should drop on
unseen drugs while barely moving on the random split, and `cyp_only` should stay
well above chance. That is the claim the paper's title needs.

Usage:
    python scripts/coldstart_ablation.py --data data/processed/features.npz
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, average_precision_score

from ddi.models.two_stage import TwoStageDDIClassifier, TwoStageConfig
from ddi import SEVERE_LABEL

MODERATE_LABEL = 1


def drug_disjoint_split(pairs, y, test_drug_frac=0.25, seed=42):
    """Hold out a fraction of DRUGS; test on pairs where both drugs are held out."""
    rng = np.random.default_rng(seed)
    drugs = sorted({d for pair in pairs for d in pair})
    rng.shuffle(drugs)
    n_test = int(len(drugs) * test_drug_frac)
    test_drugs = set(drugs[:n_test])

    test_mask = np.array([a in test_drugs and b in test_drugs for a, b in pairs])
    train_mask = np.array([a not in test_drugs and b not in test_drugs
                           for a, b in pairs])
    # Pairs straddling the boundary (one seen, one unseen) are discarded: they
    # are neither a clean cold-start case nor a clean warm one.
    return train_mask, test_mask, len(test_drugs), len(drugs)


def group_indices(names):
    idx = {n: i for i, n in enumerate(names)}
    cyp = [i for i, n in enumerate(names) if n.startswith("cyp_")]
    fp = [i for i, n in enumerate(names)
          if n.startswith("ecfp4_") or n.startswith("maccs_")]
    extra = [idx["tanimoto_sim"], idx["target_overlap_count"]]
    return cyp, fp, extra


def mask_features(X, zero_cols):
    if not zero_cols:
        return X
    Z = X.copy()
    Z[:, zero_cols] = 0.0
    return Z


def _light_config():
    """Smaller forests than the deployed bundle.

    This script fits eight models; the comparison between feature groups is
    relative, so trading a little absolute accuracy for runtime is fine. Do not
    quote these numbers as headline performance -- use scripts/train.py for that.
    """
    cfg = TwoStageConfig(use_adasyn=False)
    cfg.stage1_params = dict(cfg.stage1_params, n_estimators=150, max_depth=5)
    cfg.stage2_params = dict(cfg.stage2_params, n_estimators=150, max_depth=5)
    return cfg


def evaluate(X_tr, y_tr, X_te, y_te, tag):
    clf = TwoStageDDIClassifier(_light_config())
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)
    pred = probs.argmax(axis=1)
    out = {
        "variant": tag,
        "macro_f1": f1_score(y_te, pred, average="macro", zero_division=0),
        "severe_recall": recall_score(y_te, pred, labels=[SEVERE_LABEL],
                                      average="macro", zero_division=0),
        "moderate_f1": f1_score(y_te, pred, labels=[MODERATE_LABEL],
                                average="macro", zero_division=0),
        "severe_prauc": average_precision_score(
            (y_te == SEVERE_LABEL).astype(int), probs[:, SEVERE_LABEL]),
        "moderate_prauc": average_precision_score(
            (y_te == MODERATE_LABEL).astype(int), probs[:, MODERATE_LABEL]),
    }
    return out


def run(X, y, names, train_mask, test_mask, header):
    cyp, fp, extra = group_indices(names)
    variants = {
        "full": [],
        "no_cyp": cyp,
        "no_fingerprint": fp,
        "cyp_only": [i for i in range(X.shape[1])
                     if i not in set(cyp) | set(extra)],
    }
    X_tr_all, y_tr = X[train_mask], y[train_mask]
    X_te_all, y_te = X[test_mask], y[test_mask]

    print(f"\n=== {header} ===")
    print(f"train pairs {len(y_tr)}  test pairs {len(y_te)}")
    print(f"test class dist: {dict(zip(*np.unique(y_te, return_counts=True)))}")
    print(f"{'variant':<16}{'macroF1':>9}{'SevRec':>9}{'ModF1':>9}"
          f"{'SevPRAUC':>10}{'ModPRAUC':>10}")
    rows = []
    for tag, zero in variants.items():
        r = evaluate(mask_features(X_tr_all, zero), y_tr,
                     mask_features(X_te_all, zero), y_te, tag)
        rows.append(r)
        print(f"{r['variant']:<16}{r['macro_f1']:>9.3f}{r['severe_recall']:>9.3f}"
              f"{r['moderate_f1']:>9.3f}{r['severe_prauc']:>10.3f}"
              f"{r['moderate_prauc']:>10.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/features.npz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    blob = np.load(args.data, allow_pickle=True)
    X, y, names = blob["X"], blob["y"].astype(int), [str(n) for n in blob["names"]]
    if "pairs" not in blob:
        raise SystemExit("features.npz has no 'pairs' array; rebuild with the "
                         "current scripts/build_features.py")
    pairs = [tuple(p) for p in blob["pairs"]]

    # 1. Random pair-level split (what the paper reports).
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y,
                              random_state=args.seed)
    m_tr = np.zeros(len(y), bool); m_tr[tr] = True
    m_te = np.zeros(len(y), bool); m_te[te] = True
    warm = run(X, y, names, m_tr, m_te, "RANDOM PAIR SPLIT (drugs seen in training)")

    # 2. Drug-disjoint split (cold start).
    m_tr, m_te, n_held, n_drugs = drug_disjoint_split(pairs, y, seed=args.seed)
    print(f"\nheld out {n_held} of {n_drugs} drugs entirely")
    cold = run(X, y, names, m_tr, m_te, "DRUG-DISJOINT SPLIT (both drugs unseen)")

    print("\n=== Cost of removing each feature group (macro F1) ===")
    print(f"{'variant':<16}{'random':>10}{'cold-start':>13}{'delta':>9}")
    for w, c in zip(warm, cold):
        print(f"{w['variant']:<16}{w['macro_f1']:>10.3f}{c['macro_f1']:>13.3f}"
              f"{c['macro_f1'] - w['macro_f1']:>9.3f}")


if __name__ == "__main__":
    main()
