.PHONY: install test cov pipeline train api app clean

install:
	pip install -r requirements-dev.txt && pip install -e .

test:
	pytest tests/ -q

cov:
	pytest tests/ --cov=src/ddi --cov-report=term-missing

# Full reproducer: DrugBank XML -> labelled pairs + reference -> features -> bundle.
# Falls back to synthetic data automatically if data/raw/full_database.xml is absent.
pipeline:
	python scripts/run_pipeline.py

train:
	python scripts/train.py --out models/ddi_bundle.joblib

api:
	uvicorn api.main:app --reload

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
