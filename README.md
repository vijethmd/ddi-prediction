# Drug-Drug Interaction Prediction using Machine Learning

5-class drug-drug interaction (DDI) severity prediction: **No Interaction, Moderate, Severe, Synergistic, Antagonistic**. Built for the BMSCE Major Project, Department of Computer Science and Engineering.

The system combines a two-stage hierarchical XGBoost classifier with three research-grade enhancements — SHAP feature pruning, isotonic probability calibration, and Mondrian conformal prediction — behind a FastAPI service and a Streamlit web UI.

---

## Why this project

Around 125,000 people die each year from preventable drug-drug interactions, and 40% of elderly patients take five or more drugs at once, giving 200 million possible drug pairs to screen. Existing tools give a binary yes/no with no severity, no mechanism, and no support for newly approved drugs. This project predicts a clinically actionable severity class, explains the mechanism in plain language, and works for any drug from its SMILES string alone.

---

## What makes it different

Every one of these appears somewhere in the literature; no single deployed system combines all of them:

- **5-class severity output** instead of binary interaction detection.
- **Severe recall as a hard constraint** — a decision threshold on P(Severe) tuned to reach 0.93 recall on a held-out fold, with the precision cost reported explicitly rather than buried.
- **Cold-start support** — fingerprint + CYP450 features computed from SMILES, so novel drugs with no knowledge-graph history still get a prediction.
- **Explicit CYP450 conflict features** — inhibitor/substrate conflicts across the five major isoforms, extracted from the DrugBank enzyme records. Under DrugBank's weak-supervision labels these mark the *pharmacokinetic* (Moderate) class rather than the Severe one; see `docs/CYP_FINDINGS.md`.
- **Shared-target features** — protein-target overlap from DrugBank UniProt annotations, the pharmacodynamic counterpart to a CYP conflict.
- **Calibrated probabilities** — a reported 85% means a real-world 85%, not an overconfident tree output.
- **Conformal prediction sets** — a distribution-free coverage guarantee. Note that the sets are wide (mean size ≈ 3.7 of 5): coverage is achieved honestly, but the sets are informative only for the minority of pairs where they are small.
- **Deployed and free** — FastAPI + Streamlit, single-digit-millisecond inference on CPU once the drug is resolved offline.

---

## Repository layout

```
ddi-project/
├── src/ddi/
│   ├── data/            # DrugBank XML parser, NLP labeller, CYP knowledge base
│   ├── features/        # RDKit feature engineering, PubChem SMILES resolver
│   ├── models/          # Two-stage hierarchical XGBoost classifier
│   ├── enhance/         # SHAP pruning, isotonic calibration, Mondrian conformal
│   ├── evaluation/      # Metric suite (severe recall, macro F1, PR-AUC, kappa)
│   ├── explain/         # SHAP-to-clinical-language mechanism generator
│   └── engine.py        # Unified inference engine (model + rule fallback)
├── api/                 # FastAPI backend (POST /predict)
├── app/                 # Streamlit web UI
├── scripts/             # prepare_data, build_features, train
├── tests/               # 54 tests
├── configs/             # default.yaml hyperparameters
└── .github/workflows/   # CI (pytest on 3.10–3.12)
```

---

## Quickstart

```bash
# 1. Install
python -m pip install -r requirements-dev.txt
pip install -e .

# 2. Train a model bundle (synthetic data if no DrugBank file is supplied)
python scripts/train.py --out models/ddi_bundle.joblib

# 3. Run the API
uvicorn api.main:app --reload
#    -> http://localhost:8000/docs

# 4. Run the web UI
streamlit run app/streamlit_app.py
```

### One command to reproduce everything

```bash
make pipeline
```

This runs the whole chain — DrugBank XML → labelled pairs + SMILES/CYP reference → feature matrix → trained bundle. If `data/raw/full_database.xml` is present it uses the real offline path; otherwise it falls back to synthetic data so you always end with a working `models/ddi_bundle.joblib`.

### Predict from the API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"drug_a": "Warfarin", "drug_b": "Fluconazole"}'
```

```json
{
  "label_name": "Severe",
  "probability": 0.9,
  "cyp": "CYP2C9",
  "mechanism": "Fluconazole inhibits CYP2C9, which metabolises warfarin ...",
  "action": "Avoid or reduce warfarin dose with close INR monitoring.",
  "latency_ms": 0.04
}
```

---

## Full data pipeline (with real DrugBank)

The synthetic path lets you exercise everything without the 1.9 GB download. For the real dataset, the pipeline runs **fully offline** — SMILES strings and CYP450 profiles are extracted directly from the DrugBank XML, so no PubChem network calls are needed:

```bash
# 1. Parse DDI pairs AND extract a per-drug SMILES + CYP reference from the XML
python scripts/prepare_data.py --xml data/raw/full_database.xml
#    -> data/processed/drugbank_ddi_labelled.csv   (labelled pairs)
#    -> data/processed/drug_reference.json         (name -> SMILES + CYP450)

# 2. Build the ~3,800-feature matrix using the offline reference
python scripts/build_features.py \
    --labelled data/processed/drugbank_ddi_labelled.csv \
    --reference data/processed/drug_reference.json \
    --out data/processed/features.npz --limit 50000

# 3. Train on the real feature matrix
python scripts/train.py --data data/processed/features.npz
```

The `drug_reference.json` gives full-catalogue CYP450 coverage: `prepare_data.py`
reads each drug's `<enzymes>` block and records which CYP isoforms it inhibits or
is a substrate of, so CYP conflict features work for every drug in DrugBank — not
just a curated subset. Pass `--allow-network` to `build_features.py` only if you
want PubChem as a fallback for the rare drug with no SMILES in the XML.

---

## The three enhancements

### 1. SHAP feature pruning
Ranks features by mean absolute SHAP value and keeps the smallest set covering 95% of total importance, always retaining CYP/Tanimoto/target-overlap features. Cuts the 2,423-feature matrix to roughly 1,300 (about 45%), with no loss in Severe recall.

### 2. Isotonic calibration
Fits one isotonic regressor per class on a held-out calibration fold and renormalises, so predicted probabilities match observed frequencies. Reported by Expected Calibration Error (ECE) before/after.

### 3. Mondrian conformal prediction
Class-conditional split conformal. Computes a per-class nonconformity threshold using the finite-sample conformal quantile, and outputs a prediction *set* with a guaranteed coverage level per class — including a tighter 95% target on the Severe class.

---

## Testing

```bash
pytest tests/ --cov=src/ddi --cov-report=term-missing
```

54 tests. CI runs on Python 3.10, 3.11, and 3.12, and asserts RDKit is importable
before running the suite — an earlier CI config installed no RDKit, so the entire
structural feature path went untested.

---

## A note on metrics and honesty

The headline target is **Severe recall ≥ 0.90**, reached by thresholding P(Severe)
rather than by argmax. That threshold buys recall with precision: most pairs it
flags Severe are not severe. The trade is stated in the results, not hidden.

Three claims in the original write-up did not survive scrutiny and have been
corrected here:

- **CYP450 features were never being extracted.** `_find_cyp_profile` read
  `enzyme/gene-name`, which does not exist; the symbol lives at
  `enzyme/polypeptide/gene-name`, and the enzyme's own `<name>` reads
  "Cytochrome P450 3A4", not "CYP3A4". Every one of the 19,842 drugs got an
  empty profile, so the CYP block was 99.84% zero. Fixed.
- **CYP conflicts do not predict the Severe class.** Once extracted, a CYP
  conflict makes a pair *less* likely to be labelled Severe (9.7% vs a 24% base
  rate) and far more likely to be Moderate (88%). DrugBank's interaction text
  routes pharmacokinetic language ("the metabolism of X can be decreased…") to
  Moderate and reserves Severe for named adverse effects. See
  `docs/CYP_FINDINGS.md`.
- **`target_overlap_count` was hardcoded to `0.0`.** It is now computed from
  DrugBank UniProt target annotations.

The conformal coverage guarantee holds regardless of the underlying data, which
is why it is included — but note that it is achieved with wide sets, and a
prediction set containing all five classes is a true statement that tells you
nothing.

---

## Team

Tanmay Agarwal · Viamrsh Buyyani · Yeruva Taniya Paul · Vijeth M D
Department of Computer Science and Engineering, B.M.S. College of Engineering.

## License

MIT.
