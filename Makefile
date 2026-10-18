VENV := .venv
PY := $(VENV)/bin/python
IMAGE := heart-disease-api:latest

.PHONY: report setup data eda train test lint format docker-build docker-run k8s-deploy monitoring-up monitoring-down load-test clean

setup:  ## create venv and install dependencies
	python3.11 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt

data:  ## download and clean the dataset
	$(PY) scripts/download_data.py
	$(PY) src/data_processing.py

eda:  ## re-execute the EDA notebook
	$(VENV)/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

train:  ## train, track with MLflow and save models/
	$(PY) src/train.py

test:  ## run tests with coverage
	$(PY) -m pytest --cov=src --cov=api --cov-report=term-missing --cov-report=xml --cov-report=html

lint:  ## flake8
	$(VENV)/bin/flake8 src api scripts tests

format:
	$(VENV)/bin/black src api scripts tests

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 --name heart-api $(IMAGE)

k8s-deploy:
	bash scripts/deploy_k8s.sh

monitoring-up:
	docker compose -f monitoring/docker-compose.yml up -d --build

monitoring-down:
	docker compose -f monitoring/docker-compose.yml down

load-test:
	$(PY) scripts/load_test.py --url http://localhost:8000 --n 200

report:  ## build docs/REPORT.docx and docs/REPORT.pdf
	$(PY) scripts/build_report.py

clean:
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
