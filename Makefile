.PHONY: install data eda train test lint api docker-build docker-run

install:
	python -m pip install -r requirements.txt

data:
	PYTHONPATH=src python -m heart_ml.data_cli

eda:
	PYTHONPATH=src python -m heart_ml.eda

train:
	PYTHONPATH=src python -m heart_ml.train

test:
	PYTHONPATH=src pytest -q

lint:
	ruff check src tests scripts

api:
	PYTHONPATH=src uvicorn heart_ml.api:app --reload --port 8000

docker-build:
	docker build -t heart-mlops-api:local .

docker-run:
	docker run --rm -p 8000:8000 heart-mlops-api:local
