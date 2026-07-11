"""Unified inference engine.

Wraps the full prediction path so the FastAPI backend and the Streamlit app share
one code path.

Failure policy: when the model cannot be applied honestly -- no bundle loaded, or
a drug whose structure will not resolve -- the engine says so, via the `source`
field, rather than feeding the classifier a fabricated all-zero feature vector.
An all-zero vector is not a neutral input: the trained model maps it to
P(Severe)=0.18, which the Severe threshold promotes to a confident "Severe".
"""

import os
from dataclasses import dataclass, asdict
from typing import Optional, List

import numpy as np

from .data.knowledge_base import lookup_reference, cyp_conflicts, load_reference_cyp
from .features.pubchem import resolve_smiles, load_reference_table, reference_table
from .features.engineer import UnresolvedStructure
from .explain.explainer import explain
from . import CLASS_NAMES, CLASS_COLORS, SEVERE_LABEL


@dataclass
class Prediction:
    drug_a: str
    drug_b: str
    label: int
    label_name: str
    color: str
    probability: float
    severe_probability: Optional[float]
    promoted_by_threshold: bool
    calibrated_probs: Optional[List[float]]
    prediction_set: Optional[List[int]]
    prediction_set_names: Optional[List[str]]
    confident: Optional[bool]
    severe_in_set: bool
    mechanism: str
    action: str
    cyp: Optional[str]
    shap_drivers: Optional[List[str]]
    smiles_a: Optional[str]
    smiles_b: Optional[str]
    source: str

    def to_dict(self):
        return asdict(self)


class ModelUnavailable(RuntimeError):
    """The engine was asked for a model prediction but no bundle is loaded."""


class InferenceEngine:
    def __init__(self, model_bundle_path=None, allow_network=False,
                 reference_path=None, require_model=True):
        """Build the engine.

        allow_network defaults to False: PubChem lookups sit in the request path
        and cost seconds, and a PubChem outage would otherwise silently degrade
        every prediction. The offline DrugBank reference covers ~55k names.

        require_model=True makes a missing bundle a startup error rather than a
        silent downgrade to the 12-pair rule table.
        """
        self.allow_network = allow_network
        self.bundle = None
        self._explainer = None

        if reference_path and os.path.exists(reference_path):
            n = load_reference_table(reference_path)
            n_cyp = load_reference_cyp(reference_table())
            self.reference_size = n
            self.reference_cyp = n_cyp
        else:
            self.reference_size = 0
            self.reference_cyp = 0
            if reference_path:
                raise FileNotFoundError(
                    f"Drug reference table not found at {reference_path}. "
                    f"Run scripts/prepare_data.py to build it.")

        if model_bundle_path and os.path.exists(model_bundle_path):
            self._load_bundle(model_bundle_path)
        elif require_model:
            raise ModelUnavailable(
                f"No model bundle at {model_bundle_path!r}. Train one with "
                f"`python scripts/train.py`, or pass require_model=False to "
                f"serve the curated rule table only.")

    def _load_bundle(self, path):
        import joblib
        self.bundle = joblib.load(path)

    # -- explanation ------------------------------------------------------

    def _shap_drivers(self, vec_pruned, label):
        """Top SHAP features for this prediction, as feature names.

        Built lazily; TreeExplainer on a single row costs a few milliseconds.
        Returns None when SHAP or the model shape makes this impossible.
        """
        if self.bundle is None or label == 0:
            return None
        try:
            import shap
            if self._explainer is None:
                clf = self.bundle["classifier"]
                self._explainer = shap.TreeExplainer(clf.stage2)
            sv = self._explainer.shap_values(vec_pruned, check_additivity=False)
            sv = np.asarray(sv)
            if sv.ndim == 3:
                stage2_col = {1: 0, 2: 1, 3: 2, 4: 3}.get(label, 0)
                row = sv[0, :, stage2_col]
            else:
                row = sv[0]
            names = self._pruned_feature_names()
            if names is None or len(names) != len(row):
                return None
            order = np.argsort(np.abs(row))[::-1][:6]
            return [names[i] for i in order]
        except Exception:
            return None

    def _pruned_feature_names(self):
        names = self.bundle.get("feature_names")
        kept = self.bundle.get("kept_indices")
        if names is None:
            return None
        names = [str(n) for n in names]
        if kept is None:
            return names
        return [names[i] for i in kept]

    # -- prediction paths -------------------------------------------------

    def _rule_prediction(self, drug_a, drug_b, smiles_a, smiles_b, source=None):
        ref = lookup_reference(drug_a, drug_b)
        conflicts = cyp_conflicts(drug_a, drug_b)
        if ref is not None:
            label = ref["label"]
            prob = 0.9
        elif conflicts:
            label = SEVERE_LABEL
            prob = 0.7
        else:
            label = 0
            prob = 0.6
        ex = explain(drug_a, drug_b, label)
        if source is None:
            source = ("knowledge_base" if ref else
                      ("cyp_rule" if conflicts else "default"))
        return Prediction(
            drug_a=drug_a, drug_b=drug_b, label=label,
            label_name=CLASS_NAMES[label], color=CLASS_COLORS[label],
            probability=prob, severe_probability=None,
            promoted_by_threshold=False, calibrated_probs=None,
            prediction_set=None, prediction_set_names=None, confident=None,
            severe_in_set=(label == SEVERE_LABEL),
            mechanism=ex["mechanism"], action=ex["action"], cyp=ex.get("cyp"),
            shap_drivers=None, smiles_a=smiles_a, smiles_b=smiles_b,
            source=source,
        )

    def _model_prediction(self, drug_a, drug_b, smiles_a, smiles_b):
        from .features.engineer import featurize_pair

        # Raises UnresolvedStructure rather than fabricating zeros.
        vec = featurize_pair(smiles_a, smiles_b, drug_a, drug_b).reshape(1, -1)

        clf = self.bundle["classifier"]
        calibrator = self.bundle.get("calibrator")
        conformal = self.bundle.get("conformal")
        severe_opt = self.bundle.get("severe_threshold")
        pruner_idx = self.bundle.get("kept_indices")

        Xin = vec[:, pruner_idx] if pruner_idx is not None else vec
        raw = clf.predict_proba(Xin)
        probs = calibrator.transform(raw) if calibrator is not None else raw

        argmax_label = int(np.argmax(probs[0]))
        if severe_opt is not None:
            label = int(severe_opt.apply(probs)[0])
        else:
            label = argmax_label
        promoted = (label == SEVERE_LABEL and argmax_label != SEVERE_LABEL)

        pset = pset_names = None
        confident = None
        if conformal is not None:
            sets = conformal.predict_set(probs)
            pset = sets[0]
            pset_names = [CLASS_NAMES[c] for c in pset]
            confident = len(pset) == 1

        drivers = self._shap_drivers(Xin, label)
        ex = explain(drug_a, drug_b, label, tanimoto=float(vec[0, -2]),
                     shap_drivers=drivers,
                     target_overlap=float(vec[0, -1]))
        return Prediction(
            drug_a=drug_a, drug_b=drug_b, label=label,
            label_name=CLASS_NAMES[label], color=CLASS_COLORS[label],
            probability=float(probs[0, label]),
            severe_probability=float(probs[0, SEVERE_LABEL]),
            promoted_by_threshold=promoted,
            calibrated_probs=[round(float(p), 4) for p in probs[0]],
            prediction_set=pset, prediction_set_names=pset_names,
            confident=confident,
            severe_in_set=(pset is not None and SEVERE_LABEL in pset)
                          or label == SEVERE_LABEL,
            mechanism=ex["mechanism"], action=ex["action"], cyp=ex.get("cyp"),
            shap_drivers=drivers, smiles_a=smiles_a, smiles_b=smiles_b,
            source="model",
        )

    def predict(self, drug_a, drug_b):
        smiles_a = resolve_smiles(drug_a, self.allow_network)
        smiles_b = resolve_smiles(drug_b, self.allow_network)

        # Curated reference pairs are authoritative: prefer the knowledge base
        # over the statistical model when a pair has an authored ground truth.
        if lookup_reference(drug_a, drug_b) is not None:
            return self._rule_prediction(drug_a, drug_b, smiles_a, smiles_b)

        if self.bundle is not None:
            try:
                return self._model_prediction(drug_a, drug_b, smiles_a, smiles_b)
            except UnresolvedStructure:
                # No structure -> no fingerprints -> the model cannot be applied.
                # Fall back to the CYP rule, and label the answer as such so the
                # caller never mistakes it for a model prediction.
                return self._rule_prediction(
                    drug_a, drug_b, smiles_a, smiles_b,
                    source="rule_fallback_unresolved_structure")

        return self._rule_prediction(drug_a, drug_b, smiles_a, smiles_b)
