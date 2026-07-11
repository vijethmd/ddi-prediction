"""Streamlit web UI for the DDI checker.

Run:  streamlit run app/streamlit_app.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st

from ddi.engine import InferenceEngine
from ddi import CLASS_NAMES, CLASS_COLORS

_HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(_HERE, path)


MODEL_PATH = _abs(os.environ.get("DDI_MODEL_PATH", "models/ddi_bundle.joblib"))
REFERENCE_PATH = _abs(os.environ.get(
    "DDI_REFERENCE_PATH", "data/processed/drug_reference.json"))
ALLOW_NETWORK = os.environ.get("DDI_ALLOW_NETWORK", "").strip() == "1"

st.set_page_config(page_title="DDI Checker", page_icon="Rx", layout="centered")


@st.cache_resource
def get_engine():
    return InferenceEngine(MODEL_PATH, allow_network=ALLOW_NETWORK,
                           reference_path=REFERENCE_PATH, require_model=True)


try:
    engine = get_engine()
except Exception as exc:
    st.error(f"The prediction model could not be loaded, so this app cannot "
             f"give predictions.\n\n`{exc}`")
    st.stop()

st.title("Drug-Drug Interaction Checker")
st.caption("5-class severity prediction with calibrated confidence and "
           "conformal prediction sets. Decision support only — not a "
           "substitute for clinical judgement.")

EXAMPLES = [
    ("Warfarin", "Fluconazole"),
    ("Simvastatin", "Clarithromycin"),
    ("Trimethoprim", "Sulfamethoxazole"),
    ("Amoxicillin", "Metformin"),
    ("Clopidogrel", "Omeprazole"),
]

with st.expander("Try an example pair"):
    cols = st.columns(len(EXAMPLES))
    for col, (a, b) in zip(cols, EXAMPLES):
        if col.button(f"{a}\n+\n{b}", key=f"ex_{a}_{b}"):
            st.session_state["drug_a"] = a
            st.session_state["drug_b"] = b

c1, c2 = st.columns(2)
drug_a = c1.text_input("Drug A", key="drug_a", value=st.session_state.get("drug_a", ""))
drug_b = c2.text_input("Drug B", key="drug_b", value=st.session_state.get("drug_b", ""))

if st.button("Check interaction", type="primary"):
    if not drug_a or not drug_b:
        st.warning("Enter both drug names.")
    elif drug_a.strip().lower() == drug_b.strip().lower():
        st.warning("Enter two different drugs.")
    else:
        with st.spinner("Analysing..."):
            pred = engine.predict(drug_a, drug_b)

        color = pred.color
        st.markdown(
            f"<div style='background:{color};color:white;padding:16px 20px;"
            f"border-radius:12px;font-size:20px;font-weight:600'>"
            f"{pred.label_name} &nbsp;·&nbsp; P({pred.label_name}) = "
            f"{pred.probability:.0%}</div>", unsafe_allow_html=True)

        if pred.promoted_by_threshold:
            st.warning(
                f"Flagged Severe by the safety threshold, not by highest "
                f"probability. P(Severe) = {pred.severe_probability:.1%}, which "
                f"clears the recall-tuned cutoff. This deliberately trades "
                f"precision for recall: most pairs flagged this way are not "
                f"severe.")

        if pred.source.startswith("rule_fallback"):
            st.error(
                "Could not resolve a molecular structure for one of these drugs, "
                "so the model was not applied. The result below comes from the "
                "CYP450 rule table only.")

        st.subheader("Mechanism")
        st.write(pred.mechanism)
        st.subheader("Recommended action")
        st.write(pred.action)

        if pred.cyp:
            st.info(f"CYP450 pathway involved: {pred.cyp}")

        if pred.shap_drivers:
            st.caption("Top model drivers: " + ", ".join(pred.shap_drivers[:4]))

        if pred.prediction_set_names:
            st.subheader("Conformal prediction set")
            if pred.confident:
                st.success(f"Confident single label: {pred.prediction_set_names[0]}")
            else:
                st.warning(
                    "Model is uncertain. The guarantee says the true class is "
                    "among: " + ", ".join(pred.prediction_set_names))

        if pred.calibrated_probs:
            st.subheader("Calibrated probabilities")
            for i, p in enumerate(pred.calibrated_probs):
                st.write(f"{CLASS_NAMES[i]}")
                st.progress(min(max(p, 0.0), 1.0))

        with st.expander("Molecular details"):
            st.write(f"SMILES A: `{pred.smiles_a or 'not resolved'}`")
            st.write(f"SMILES B: `{pred.smiles_b or 'not resolved'}`")
            st.write(f"Prediction source: {pred.source}")

st.divider()
st.caption("BMSCE Major Project · Department of CSE · Drug-Drug Interaction "
           "Prediction using Machine Learning")
