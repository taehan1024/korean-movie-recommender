VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: setup ingest ingest-us ingest-kr fetch-keywords features features-quick \
       eval eval-tune app quality results test clean

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	@echo "Done. Activate with: source $(VENV)/bin/activate"
	@echo "Set TMDB_API_KEY in .env before running ingest"

ingest:
	$(PYTHON) data_ingestion.py

ingest-us:
	$(PYTHON) data_ingestion.py --skip-kr

ingest-kr:
	$(PYTHON) data_ingestion.py --skip-us

fetch-keywords:
	$(PYTHON) fetch_keywords.py

features:
	$(PYTHON) feature_engineering.py --groups tfidf,embedding,genre,cast,keyword,year

features-quick:
	$(PYTHON) feature_engineering.py --groups tfidf,genre

eval:
	$(PYTHON) evaluate.py --model all

eval-tune:
	$(PYTHON) evaluate.py --tune

app:
	$(PYTHON) -m streamlit run streamlit_app.py

quality:
	@echo "--- Data Quality ---"
	@$(PYTHON) -c "import pandas as pd, json; \
		us = pd.read_csv('data/processed/us_movies.csv'); \
		kr = pd.read_csv('data/processed/kr_movies.csv'); \
		print(f'US movies: {len(us)}'); \
		print(f'KR movies: {len(kr)}'); \
		print(f'US overview missing: {us.overview.isna().mean():.1%}'); \
		print(f'KR overview missing: {kr.overview.isna().mean():.1%}'); \
		print(f'US director missing: {us.director.isna().mean():.1%}'); \
		print(f'KR director missing: {kr.director.isna().mean():.1%}')"

results:
	@cat results/benchmark_comparison.csv

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf data/raw data/features data/metadata results
