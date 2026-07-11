import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ddi.data.nlp_labeller import classify_description
from ddi.data.knowledge_base import lookup_reference, cyp_conflicts
from ddi import CLASS_INDEX


def test_severe_keyword():
    assert classify_description(
        "This combination is contraindicated.") == CLASS_INDEX["Severe"]


def test_severe_precedence_over_moderate():
    # Contains both 'monitor' and 'contraindicated' -> Severe wins.
    assert classify_description(
        "Monitor closely; the combination is contraindicated."
    ) == CLASS_INDEX["Severe"]


def test_synergistic_keyword():
    assert classify_description(
        "Drug A potentiates the effect of Drug B.") == CLASS_INDEX["Synergistic"]


def test_antagonistic_keyword():
    assert classify_description(
        "Drug A reduces efficacy of Drug B.") == CLASS_INDEX["Antagonistic"]


def test_moderate_keyword():
    assert classify_description(
        "Monitor the patient for increased risk.") == CLASS_INDEX["Moderate"]


def test_no_interaction_empty():
    assert classify_description("") == CLASS_INDEX["No Interaction"]
    assert classify_description(
        "These agents are chemically unrelated.") == CLASS_INDEX["No Interaction"]


def test_reference_lookup_symmetric():
    a = lookup_reference("Warfarin", "Fluconazole")
    b = lookup_reference("Fluconazole", "Warfarin")
    assert a is not None and b is not None
    assert a["label"] == b["label"] == 2


def test_cyp_conflict_detection():
    conflicts = cyp_conflicts("warfarin", "fluconazole")
    isos = {c[0] for c in conflicts}
    assert "CYP2C9" in isos


def test_cyp_no_conflict_unknown():
    assert cyp_conflicts("amoxicillin", "metformin") == []
