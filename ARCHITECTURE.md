# Architecture

This document describes how the DDI prediction system fits together, from raw data to a served prediction.

## Overview

The system has four layers:

1. **Data layer** — parses DrugBank, generates 5-class labels, and holds the CYP450 knowledge base.
2. **Feature layer** — resolves drug names to SMILES and computes the 2,423-feature vector per pair.
3. **Model layer** — the two-stage classifier plus the three enhancement wrappers.
4. **Serving layer** — a unified inference engine behind a FastAPI service and a Streamlit UI.

## Data layer (`src/ddi/data/`)

- `parse_drugbank.py` streams the 1.9 GB DrugBank XML with `iterparse`, clearing each element after use so memory stays flat. It provides two extractions: `parse_drugbank()` writes one CSV row per drug-interaction pair, and `extract_drug_reference()` writes a per-drug JSON of SMILES strings, CYP450 profiles, protein targets, and synonyms. CYP isoforms are read from `enzyme/polypeptide/gene-name` (with a long-form-name fallback); an earlier version read a non-existent `enzyme/gene-name` field and silently produced empty profiles for all drugs. The reference JSON is what makes the whole training pipeline run offline — SMILES, CYP, and target data come from the XML rather than the PubChem network API.
- `nlp_labeller.py` assigns one of five severity classes from the free-text interaction description using an ordered keyword ruleset (Severe checked before Moderate, etc.).
- `knowledge_base.py` holds per-drug CYP450 inhibitor/substrate profiles, curated reference pairs with authored mechanisms, and pharmacodynamic rules. This is authoritative ground truth used both for features and for the deployed rule fallback.

## Feature layer (`src/ddi/features/`)

- `pubchem.py` resolves a drug name to a canonical SMILES string, with an offline table for the reference drugs and an LRU cache.
- `engineer.py` computes the full feature vector: ECFP4 Morgan fingerprints (2×1024), MACCS keys (2×167), seven physicochemical descriptors per drug, 25 CYP450 flags (inhibitor/substrate/conflict across five isoforms), Tanimoto similarity, and a shared-target count. The pair is canonically ordered so `f(A,B) == f(B,A)`. RDKit is required, not optional: a missing structure raises `UnresolvedStructure` rather than returning an all-zero vector, because the model maps all-zeros to a confident (and wrong) Severe.

## Model layer

### Two-stage classifier (`src/ddi/models/two_stage.py`)

- **Stage 1 (safety gate):** binary XGBoost, decision threshold 0.30. Any pair scoring ≥0.30 is treated as interacting and passed on. The low threshold maximises recall of the interacting classes.
- **Stage 2 (type classifier):** 4-class XGBoost over the interacting classes, with sample weights emphasising Severe (5.0) and ADASYN oversampling of minority classes.
- `predict_proba` composes a full 5-class distribution: `P(no interaction) = 1 - gate`, and `P(type) = gate × stage2_softmax(type)`.

### Enhancement layer (`src/ddi/enhance/`)

- `pruning.py` — SHAP-based feature selection with protected features.
- `calibration.py` — per-class isotonic regression with renormalisation.
- `conformal.py` — Mondrian (class-conditional) split conformal with LAC or APS nonconformity scores.
- `pipeline.py` — `EnhancedDDIClassifier` orchestrates all three with leak-free data splits (model / calibration / conformal / test).

## Serving layer

- `engine.py` — `InferenceEngine` is the single prediction path shared by the API and UI. Curated reference pairs are answered from the knowledge base (authoritative); everything else goes through the trained bundle; if no bundle is present, a CYP rule fallback responds.
- `api/main.py` — FastAPI `POST /predict` with Pydantic validation, CORS, and a latency field.
- `app/streamlit_app.py` — severity badge, mechanism, action, conformal set display, and calibrated probability bars.

## Split discipline

To avoid leakage, the training script partitions data four ways:

- **model** — trains the two-stage XGBoost.
- **calibration** — fits isotonic calibration (the base model never sees it).
- **conformal** — sets conformal thresholds using calibrated probabilities.
- **test** — final held-out evaluation only.

## Prediction flow

```
drug names
   → offline DrugBank SMILES resolution (PubChem only if enabled)
   → RDKit feature vector (2,423 dims, pruned to 1,339)
   → two-stage XGBoost  → raw 5-class probabilities
   → isotonic calibration → calibrated probabilities
   → Mondrian conformal   → prediction set + guarantee
   → SHAP / knowledge base → mechanism + action
   → JSON response
```
