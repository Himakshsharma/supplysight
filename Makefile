# SupplySight – one-command workflows (Windows users: use `python run.py <task>`)
PY ?= python

.PHONY: setup all data train drift retrain eval test lint format api dashboard mlflow up down clean

setup:            ## install the package with API/dashboard extras
	$(PY) -m pip install -e ".[serve,dev]"

all: data train drift eval   ## rebuild data, train models, drift report, evaluations

data:
	$(PY) scripts/generate_data.py --seed 42

train:
	$(PY) scripts/train_all.py

drift:
	$(PY) scripts/simulate_drift.py
	$(PY) scripts/retrain.py --check

retrain:
	$(PY) scripts/retrain.py

eval:
	$(PY) scripts/evaluate_all.py
	$(PY) scripts/make_eda_charts.py

test:
	$(PY) -m pytest --cov=src/supplysight --cov-report=term-missing

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

api:
	$(PY) -m uvicorn supplysight.api.main:app --reload --port 8000

dashboard:
	$(PY) -m streamlit run dashboard/app.py

mlflow:
	$(PY) -m mlflow ui --port 5000

up:               ## postgres + mlflow + trainer + api + dashboard
	docker compose up --build

down:
	docker compose down -v

clean:
	rm -rf data/raw data/drift data/*.db artifacts/* .pytest_cache .coverage
	touch artifacts/.gitkeep
