# Deployment

The repo ships the trained model bundle (`models/ddi_bundle.joblib`) and the
drug reference (`data/processed/drug_reference.json`), so both targets below
deploy straight from a clone with no data-build step.

## Streamlit Cloud (the demo UI — drugdrug.streamlit.app)

Streamlit Cloud builds from this repo directly:

1. share.streamlit.io -> New app -> pick this repo/branch.
2. Main file path: `app/streamlit_app.py`.
3. Deploy. Streamlit Cloud installs `requirements.txt` (which now includes
   RDKit) and the apt packages in `packages.txt` (`libxrender1`, `libxext6`,
   `libgomp1`, needed by RDKit and XGBoost). `.streamlit/config.toml` sets the
   theme and server flags.

No secrets are required — the app runs fully offline against the bundled
reference table. `DDI_ALLOW_NETWORK` is off by default; set it in the app's
Secrets only if you want PubChem fallback for drugs missing from the reference.

## Render (the REST API)

`render.yaml` is a Blueprint that builds the `Dockerfile` (which installs the
RDKit system libs and bakes in the bundle + reference):

1. dashboard.render.com -> New -> Blueprint -> point at this repo.
2. Render reads `render.yaml`, builds the image, and serves `uvicorn` on the
   port it injects. `/health` is the health check.
3. Update `DDI_CORS_ORIGINS` to the real Streamlit URL once it is live.

Local equivalent:

    docker build -t ddi-api .
    docker run -p 8000:8000 ddi-api
    # -> http://localhost:8000/docs

## Regenerating the artifacts

The committed bundle and reference are enough to serve. To rebuild them from
scratch you need the 1.9 GB DrugBank XML at `data/raw/full_database.xml`, then:

    make pipeline        # XML -> labelled pairs + reference -> features -> bundle
    python3 make_figures.py
