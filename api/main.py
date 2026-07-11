"""FastAPI backend for DDI prediction.

Run:  uvicorn api.main:app --reload
Docs: http://localhost:8000/docs

Configuration (environment):
    DDI_MODEL_PATH      path to the joblib bundle      (default: models/ddi_bundle.joblib)
    DDI_REFERENCE_PATH  path to drug_reference.json    (default: data/processed/drug_reference.json)
    DDI_ALLOW_NETWORK   "1" to permit PubChem fallback (default: off)
    DDI_CORS_ORIGINS    comma-separated allowed origins (default: localhost only)
"""

import os
import sys
import time

_HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_HERE, "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ddi.engine import InferenceEngine
from ddi import CLASS_NAMES


def _abs(path):
    """Resolve a path against the repo root, not the process CWD.

    The default MODEL_PATH used to be relative, so launching uvicorn from any
    other directory silently produced a model-less engine.
    """
    return path if os.path.isabs(path) else os.path.join(_HERE, path)


MODEL_PATH = _abs(os.environ.get("DDI_MODEL_PATH", "models/ddi_bundle.joblib"))
REFERENCE_PATH = _abs(os.environ.get(
    "DDI_REFERENCE_PATH", "data/processed/drug_reference.json"))
ALLOW_NETWORK = os.environ.get("DDI_ALLOW_NETWORK", "").strip() == "1"
CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "DDI_CORS_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",")
    if o.strip()]

app = FastAPI(
    title="DDI Prediction API",
    description="5-class drug-drug interaction severity prediction.",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Raises at import if the bundle or reference table is missing. A medical
# decision-support endpoint that boots without its model and answers anyway is
# worse than one that refuses to boot.
engine = InferenceEngine(
    MODEL_PATH, allow_network=ALLOW_NETWORK,
    reference_path=REFERENCE_PATH, require_model=True,
)


class PredictRequest(BaseModel):
    drug_a: str = Field(..., min_length=1, max_length=200, examples=["Warfarin"])
    drug_b: str = Field(..., min_length=1, max_length=200, examples=["Fluconazole"])


class PredictResponse(BaseModel):
    drug_a: str
    drug_b: str
    label: int
    label_name: str
    color: str
    probability: float
    severe_probability: float | None
    promoted_by_threshold: bool
    calibrated_probs: list | None
    prediction_set: list | None
    prediction_set_names: list | None
    confident: bool | None
    severe_in_set: bool
    mechanism: str
    action: str
    cyp: str | None
    shap_drivers: list | None
    smiles_a: str | None
    smiles_b: str | None
    source: str
    latency_ms: float


@app.get("/")
def root():
    return {
        "service": "DDI Prediction API",
        "version": "1.1.0",
        "model_loaded": engine.bundle is not None,
        "reference_drugs": engine.reference_size,
        "reference_drugs_with_cyp": engine.reference_cyp,
        "network_fallback": ALLOW_NETWORK,
        "classes": CLASS_NAMES,
    }


@app.get("/health")
def health():
    if engine.bundle is None:
        raise HTTPException(503, "model bundle not loaded")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.drug_a.strip().lower() == req.drug_b.strip().lower():
        raise HTTPException(400, "Please provide two different drugs.")
    t0 = time.perf_counter()
    pred = engine.predict(req.drug_a, req.drug_b)
    latency = (time.perf_counter() - t0) * 1000.0
    payload = pred.to_dict()
    payload["latency_ms"] = round(latency, 2)
    return payload
