"""Train the full DDI model bundle: two-stage classifier + SHAP pruning +
isotonic calibration + Mondrian conformal, then serialise for deployment.

Usage:
    python scripts/train.py --data data/processed/features.npz --out models/ddi_bundle.joblib

If --data is omitted, a synthetic DDI-shaped dataset is generated so the full
training path can be exercised without the 1.9 GB DrugBank download.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import joblib

from ddi.models.two_stage import TwoStageDDIClassifier, TwoStageConfig
from ddi.enhance import (
    ShapFeaturePruner, IsotonicCalibrator, MondrianConformalPredictor,
    calibration_report,
)
from ddi.evaluation.report import full_report, print_report
from ddi import CLASS_NAMES, SEVERE_LABEL


def synthetic_dataset(n=8000, d=300, seed=42):
    rng = np.random.default_rng(seed)
    n_fp = d - 30
    fp = (rng.random((n, n_fp)) < 0.06).astype(float)
    physchem = rng.normal(0, 1, (n, 20))
    cyp = (rng.random((n, 8)) < 0.15).astype(float)
    tan = rng.random((n, 1))
    tgt = (rng.random((n, 1)) < 0.2).astype(float)
    X = np.hstack([fp, physchem, cyp, tan, tgt])

    names = [f"ecfp4_bit_{i}" for i in range(n_fp)]
    names += [f"physchem_{i}" for i in range(20)]
    names += [f"cyp_{iso}_{role}" for iso in ["3a4", "2c9", "2d6", "2c19"]
              for role in ["conflict", "substrate"]]
    names += ["tanimoto_sim", "target_overlap_count"]

    load = cyp[:, [0, 2, 4, 6]].sum(axis=1)
    severe = 3.2 * load + 2.2 * tan[:, 0] + 1.1 * tgt[:, 0] + rng.normal(0, 0.4, n)
    moderate = 1.6 * np.abs(physchem[:, 0]) + 1.2 * cyp[:, 1] + rng.normal(0, 0.5, n)
    syn = 2.6 * tgt[:, 0] + 1.0 * tan[:, 0] + rng.normal(0, 0.4, n)
    antag = 1.8 * np.abs(physchem[:, 3]) - 1.0 * tan[:, 0] + rng.normal(0, 0.4, n)
    noint = np.full(n, 1.0) + rng.normal(0, 0.4, n)
    logits = np.vstack([noint, moderate, severe, syn, antag]).T
    logits += np.array([3.0, 1.4, -1.6, -2.0, -1.7])
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(5, p=p[i]) for i in range(n)])
    return X, y, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="models/ddi_bundle.joblib")
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--coverage", type=float, default=0.95)
    args = ap.parse_args()

    if args.data and os.path.exists(args.data):
        blob = np.load(args.data, allow_pickle=True)
        X, y, names = blob["X"], blob["y"], list(blob["names"])
        print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features from {args.data}")
    else:
        print("No data file given — generating synthetic DDI-shaped dataset.")
        X, y, names = synthetic_dataset()

    from sklearn.model_selection import train_test_split
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_model, X_hold, y_model, y_hold = train_test_split(
        X_tmp, y_tmp, test_size=0.3, stratify=y_tmp, random_state=42)
    X_calib, X_conf, y_calib, y_conf = train_test_split(
        X_hold, y_hold, test_size=0.5, stratify=y_hold, random_state=42)

    print(f"Splits  model={len(y_model)} calib={len(y_calib)} "
          f"conf={len(y_conf)} test={len(y_test)}")

    print("\n[1/4] Training two-stage classifier (full features)...")
    t0 = time.perf_counter()
    clf_full = TwoStageDDIClassifier(TwoStageConfig(use_adasyn=True))
    clf_full.fit(X_model, y_model)
    print(f"  trained in {time.perf_counter()-t0:.1f}s")

    kept_indices = None
    clf = clf_full
    X_model_used, X_calib_used = X_model, X_calib
    X_conf_used, X_test_used = X_conf, X_test
    if not args.no_prune:
        print("\n[2/4] SHAP feature pruning...")
        pruner = ShapFeaturePruner(coverage=args.coverage)
        import xgboost as xgb
        proxy = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, tree_method="hist",
            n_jobs=-1, eval_metric="mlogloss", random_state=42)
        proxy.fit(X_model, y_model)
        pr = pruner.fit(proxy, X_model, names)
        kept_indices = pr.kept_indices
        print("  " + pr.summary())
        print("  retraining two-stage on pruned features for a consistent bundle...")
        X_model_used = X_model[:, kept_indices]
        X_calib_used = X_calib[:, kept_indices]
        X_conf_used = X_conf[:, kept_indices]
        X_test_used = X_test[:, kept_indices]
        clf = TwoStageDDIClassifier(TwoStageConfig(use_adasyn=True))
        clf.fit(X_model_used, y_model)
    else:
        print("\n[2/4] Pruning skipped.")

    print("\n[3/4] Isotonic calibration...")
    raw_calib = clf.predict_proba(X_calib_used)
    calibrator = IsotonicCalibrator(n_classes=5).fit(raw_calib, y_calib)
    raw_test = clf.predict_proba(X_test_used)
    cal_test = calibrator.transform(raw_test)
    rep = calibration_report(y_test, raw_test, cal_test, 5)
    print(rep.summary(CLASS_NAMES))

    print("\n[4/4] Mondrian conformal calibration...")
    raw_conf = clf.predict_proba(X_conf_used)
    cal_conf = calibrator.transform(raw_conf)
    conformal = MondrianConformalPredictor(
        n_classes=5, alpha_per_class={0: 0.1, 1: 0.1, 2: 0.05, 3: 0.1, 4: 0.1},
        score="aps")
    conformal.calibrate(cal_conf, y_conf)
    cres = conformal.evaluate(cal_test, y_test)
    print(cres.summary(CLASS_NAMES))

    print("\n[+] Severe-recall threshold optimisation...")
    from ddi.enhance import SevereThresholdOptimizer
    # Fit on the calibration fold's calibrated probs; target slightly above 0.90
    # to absorb calibration->test variance so test recall clears the constraint.
    opt = SevereThresholdOptimizer(target_recall=0.93)
    opt.fit(calibrator.transform(clf.predict_proba(X_conf_used)), y_conf)
    print("  " + opt.result_.summary())
    tr = opt.tradeoff(cal_test, y_test)
    print(f"  Severe recall: {tr['severe_recall_plain']:.3f} -> "
          f"{tr['severe_recall_thresholded']:.3f}  "
          f"(macro F1 {tr['macro_f1_plain']:.3f} -> "
          f"{tr['macro_f1_thresholded']:.3f})")

    print("\nFinal test-set classification report (with Severe threshold):")
    preds = opt.apply(cal_test)
    print_report(full_report(y_test, preds, cal_test))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({
        "classifier": clf,
        "calibrator": calibrator,
        "conformal": conformal,
        "severe_threshold": opt,
        "kept_indices": kept_indices,
        "feature_names": names,
        "class_names": CLASS_NAMES,
    }, args.out)
    print(f"\nSaved model bundle -> {args.out}")


if __name__ == "__main__":
    main()
