import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import xgboost as xgb

from ddi.enhance.pruning import ShapFeaturePruner
from ddi.enhance.pipeline import EnhancedDDIClassifier, PipelineConfig
from ddi.explain.explainer import explain
from ddi.evaluation.report import full_report, print_report
from ddi.enhance.metrics import reliability_curve


def _learnable(n=800, d=60, seed=2):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    score = 1.5 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2]
    y = np.zeros(n, dtype=int)
    y[score > 0.6] = 1
    y[score > 1.4] = 2
    y[score < -1.0] = 3
    y[(score < -0.3) & (score >= -1.0)] = 4
    names = [f"cyp_feat_{i}" if i < 3 else f"bit_{i}" for i in range(d)]
    return X, y, names


def _factory():
    return xgb.XGBClassifier(
        n_estimators=120, max_depth=4, tree_method="hist",
        n_jobs=1, eval_metric="mlogloss", random_state=42)


def test_pruner_reduces_and_protects():
    X, y, names = _learnable()
    model = _factory().fit(X, y)
    pruner = ShapFeaturePruner(coverage=0.9,
                               protected_substrings=("cyp",))
    res = pruner.fit(model, X, names)
    assert res.n_kept <= res.n_full
    # All protected cyp features retained.
    assert all(any(p in kn for kn in res.kept_names)
               for p in ["cyp_feat_0", "cyp_feat_1", "cyp_feat_2"])
    assert 0.0 < res.reduction_pct <= 100.0
    assert res.summary()


def test_pruner_transform_shape():
    X, y, names = _learnable()
    model = _factory().fit(X, y)
    res = ShapFeaturePruner(coverage=0.9).fit(model, X, names)
    Xt = ShapFeaturePruner.transform(X, res)
    assert Xt.shape[1] == res.n_kept


def test_pruner_top_k_table():
    X, y, names = _learnable()
    model = _factory().fit(X, y)
    pruner = ShapFeaturePruner(coverage=0.95)
    res = pruner.fit(model, X, names)
    table = pruner.top_k_table(res, k=5)
    assert len(table) == 5
    assert table[0]["rank"] == 1


def test_enhanced_pipeline_end_to_end():
    X, y, names = _learnable(n=1200)
    clf = EnhancedDDIClassifier(_factory, PipelineConfig(coverage=0.9))
    clf.fit(X, y, names, prune=True, verbose=False)
    ev = clf.evaluate_test()
    assert "severe_recall_pruned" in ev
    preds = clf.predict(X[:20])
    assert len(preds) == 20
    for p in preds:
        assert p.label in range(5)
        assert len(p.prediction_set) >= 1
        d = p.as_dict()
        assert "prediction_set_names" in d


def test_explain_knowledge_base_path():
    ex = explain("warfarin", "fluconazole", 2)
    assert ex["source"] == "knowledge_base"
    assert ex["cyp"] == "CYP2C9"


def test_explain_heuristic_path():
    ex = explain("unknowndrugA", "unknowndrugB", 1)
    assert ex["mechanism"]
    assert ex["action"]


def test_explain_high_tanimoto():
    ex = explain("unknownA", "unknownB", 2, tanimoto=0.8)
    assert "similar" in ex["mechanism"].lower()


def test_full_report_and_print(capsys):
    y_true = np.array([0, 1, 2, 2, 0, 1, 2, 3, 4, 0])
    y_pred = np.array([0, 1, 2, 0, 0, 1, 2, 3, 4, 1])
    probs = np.eye(5)[y_pred]
    rep = full_report(y_true, y_pred, probs)
    assert "severe_recall" in rep
    assert "pr_auc_per_class" in rep
    print_report(rep)
    out = capsys.readouterr().out
    assert "Severe recall" in out


def test_reliability_curve_shape():
    rng = np.random.default_rng(0)
    p = rng.random(200)
    y = (rng.random(200) < p).astype(float)
    rc = reliability_curve(y, p, n_bins=10)
    assert len(rc["bin_centers"]) == 10
    assert len(rc["accuracy"]) == 10
