"""Mechanism explanation generator.

Maps model drivers to plain-language clinical text. Two paths:
  1. Knowledge-base path: if the pair is a curated reference pair, return its
     authored mechanism and action (highest fidelity).
  2. SHAP path: rank feature contributions for the predicted class and translate
     the top features (CYP conflicts, Tanimoto, target overlap) to text.
"""

import numpy as np

from ..data.knowledge_base import lookup_reference, cyp_conflicts
from .. import CLASS_NAMES, SEVERE_LABEL


def _cyp_sentence(conflicts):
    if not conflicts:
        return None
    iso, inhibitor, substrate = conflicts[0]
    return (f"{inhibitor.title()} inhibits {iso}, which metabolises "
            f"{substrate.title()}. {substrate.title()} levels may rise, "
            f"increasing the risk of adverse effects.")


def explain(drug_a, drug_b, predicted_label, shap_drivers=None,
            tanimoto=None, target_overlap=None):
    """Return a dict with mechanism text and clinical action.

    shap_drivers is the ordered list of top feature names for this prediction,
    computed by the engine. A CYP conflict feature appearing among them is what
    licenses the pharmacokinetic mechanism sentence.
    """
    ref = lookup_reference(drug_a, drug_b)
    if ref is not None:
        return {
            "source": "knowledge_base",
            "mechanism": ref["mechanism"],
            "action": ref["action"],
            "cyp": ref.get("cyp"),
        }

    conflicts = cyp_conflicts(drug_a, drug_b)
    cyp_text = _cyp_sentence(conflicts)

    drivers = [d for d in (shap_drivers or [])
               if "cyp" in d.lower() and "conflict" in d.lower()]

    label_name = CLASS_NAMES.get(predicted_label, str(predicted_label))
    if cyp_text:
        mechanism = cyp_text
        if drivers:
            mechanism += (f" This pathway is also the model's top driver for "
                          f"the prediction ({drivers[0]}).")
    elif target_overlap:
        n = int(target_overlap)
        mechanism = (f"{drug_a.title()} and {drug_b.title()} act on "
                     f"{n} shared protein target{'s' if n != 1 else ''}, so "
                     f"their effects combine directly without either changing "
                     f"the other's concentration.")
    elif tanimoto is not None and tanimoto > 0.6:
        mechanism = (f"{drug_a.title()} and {drug_b.title()} are structurally "
                     f"similar (Tanimoto {tanimoto:.2f}) and may compete for the "
                     f"same binding sites or metabolic pathways.")
    elif predicted_label == 0:
        mechanism = (f"No specific interaction mechanism was identified for "
                     f"{drug_a.title()} and {drug_b.title()} from the available "
                     f"molecular and pharmacological features.")
    else:
        mechanism = (f"A {label_name.lower()} interaction is indicated by the "
                     f"combined molecular and pharmacological feature profile "
                     f"of the two drugs.")
        if shap_drivers:
            mechanism += (f" Top contributing features: "
                          f"{', '.join(shap_drivers[:3])}.")

    if predicted_label == SEVERE_LABEL:
        action = "Avoid the combination or use only with close monitoring."
    elif predicted_label == 1:
        action = "Monitor the patient; dose adjustment may be required."
    elif predicted_label == 3:
        action = "Beneficial combination — may be used intentionally."
    elif predicted_label == 4:
        action = "One drug may reduce the other's efficacy; review the regimen."
    else:
        action = "No specific action required."

    return {
        "source": "shap" if drivers else "heuristic",
        "mechanism": mechanism,
        "action": action,
        "cyp": conflicts[0][0] if conflicts else None,
        "shap_drivers": shap_drivers or [],
    }
