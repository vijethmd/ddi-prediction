import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

import pytest

from ddi.models.two_stage import TwoStageDDIClassifier, TwoStageConfig
from ddi.features.engineer import (
    feature_names, N_FEATURES, featurize_pair, UnresolvedStructure,
)
from ddi.features.pubchem import resolve_smiles
from ddi.engine import InferenceEngine, ModelUnavailable


def _toy_dataset(n=600, d=40, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    # Class driven by first two features so it's learnable.
    score = X[:, 0] + 0.5 * X[:, 1]
    y = np.zeros(n, dtype=int)
    y[score > 0.5] = 1
    y[score > 1.2] = 2
    y[score < -0.8] = 3
    y[(score < -0.2) & (score >= -0.8)] = 4
    return X, y


def test_two_stage_fit_predict_shapes():
    X, y = _toy_dataset()
    clf = TwoStageDDIClassifier(TwoStageConfig(use_adasyn=False))
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 5)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    preds = clf.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1, 2, 3, 4})


def test_two_stage_save_load(tmp_path):
    X, y = _toy_dataset()
    clf = TwoStageDDIClassifier(TwoStageConfig(use_adasyn=False)).fit(X, y)
    path = tmp_path / "clf.joblib"
    clf.save(str(path))
    loaded = TwoStageDDIClassifier.load(str(path))
    assert np.array_equal(clf.predict(X), loaded.predict(X))


def test_feature_names_match_count():
    assert len(feature_names()) == N_FEATURES


def _smiles(name):
    return resolve_smiles(name, allow_network=False)


def test_featurize_pair_length():
    vec = featurize_pair(_smiles("warfarin"), _smiles("fluconazole"),
                         "warfarin", "fluconazole")
    assert vec.shape[0] == N_FEATURES


def test_featurize_raises_on_unresolved_smiles():
    """An all-zero structural block must never be fabricated: the model maps it
    to a confident Severe. Missing structure has to be an error."""
    with pytest.raises(UnresolvedStructure):
        featurize_pair(None, None, "warfarin", "fluconazole")
    with pytest.raises(UnresolvedStructure):
        featurize_pair(_smiles("warfarin"), None, "warfarin", "mystery_drug")


def test_featurize_raises_on_unparseable_smiles():
    with pytest.raises(UnresolvedStructure):
        featurize_pair("not-a-smiles((", _smiles("warfarin"), "x", "warfarin")


def test_featurize_cyp_conflict_flag_set():
    names = feature_names()
    vec = featurize_pair(_smiles("warfarin"), _smiles("fluconazole"),
                         "warfarin", "fluconazole")
    idx = names.index("cyp_CYP2C9_conflict")
    assert vec[idx] == 1.0


def test_featurize_is_order_invariant():
    """A drug interaction is symmetric; the feature vector must be too."""
    ab = featurize_pair(_smiles("warfarin"), _smiles("fluconazole"),
                        "warfarin", "fluconazole")
    ba = featurize_pair(_smiles("fluconazole"), _smiles("warfarin"),
                        "fluconazole", "warfarin")
    assert np.array_equal(ab, ba)


def test_offline_smiles_resolution():
    assert resolve_smiles("warfarin", allow_network=False) is not None
    assert resolve_smiles("nonexistent_drug_xyz", allow_network=False) is None


def test_engine_requires_model_by_default():
    """A missing bundle must be a startup error, not a silent downgrade."""
    with pytest.raises(ModelUnavailable):
        InferenceEngine(model_bundle_path=None, allow_network=False)


def test_engine_rule_fallback():
    eng = InferenceEngine(model_bundle_path=None, allow_network=False,
                          require_model=False)
    pred = eng.predict("Warfarin", "Fluconazole")
    assert pred.label == 2
    assert pred.severe_in_set
    assert "CYP2C9" == pred.cyp


def test_engine_no_interaction_pair():
    eng = InferenceEngine(model_bundle_path=None, allow_network=False,
                          require_model=False)
    pred = eng.predict("Amoxicillin", "Metformin")
    assert pred.label == 0
