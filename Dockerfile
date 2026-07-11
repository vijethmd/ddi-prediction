# RDKit needs libXrender/libXext at runtime (the same libs packages.txt lists
# for Streamlit Cloud). Installing them here is what keeps the structural
# feature path alive in a container.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DDI_MODEL_PATH=/app/models/ddi_bundle.joblib \
    DDI_REFERENCE_PATH=/app/data/processed/drug_reference.json

RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY app/ ./app/
COPY configs/ ./configs/

# The bundle and the drug reference are build inputs, not code. Generate them
# with `make pipeline` and mount or COPY them in; the API refuses to start
# without both, by design.
COPY models/ ./models/
COPY data/processed/drug_reference.json ./data/processed/drug_reference.json

RUN pip install --no-cache-dir -e . \
    && python -c "from ddi.features.engineer import RDKIT_AVAILABLE; assert RDKIT_AVAILABLE"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').status_code==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
