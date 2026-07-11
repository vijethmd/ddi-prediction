"""Drug-Drug Interaction Prediction — 5-class severity classification.

BMSCE Major Project, Department of CSE.
"""

__version__ = "1.0.0"

CLASS_NAMES = {
    0: "No Interaction",
    1: "Moderate",
    2: "Severe",
    3: "Synergistic",
    4: "Antagonistic",
}

CLASS_INDEX = {v: k for k, v in CLASS_NAMES.items()}

SEVERE_LABEL = 2

# Colour codes used by the UI layer for each severity class.
CLASS_COLORS = {
    0: "#3B6D11",  # green  — No Interaction
    1: "#BA7517",  # amber  — Moderate
    2: "#A32D2D",  # red    — Severe
    3: "#185FA5",  # blue   — Synergistic
    4: "#534AB7",  # purple — Antagonistic
}

CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4"]
