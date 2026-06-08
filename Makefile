.PHONY: install up down fmt lint test doctor train demo clean

install:  ## uv sync with dev extras
	uv sync --extra dev

up:  ## bring up MLflow on :5050
	docker compose up -d

down:  ## stop services (volumes preserved)
	docker compose down

fmt:  ## ruff format
	uv run ruff format src tests

lint:  ## ruff check
	uv run ruff check src tests

test:  ## pytest smoke suite
	uv run pytest

doctor:  ## environment readiness check
	uv run mlt doctor

train:  ## train the reference config
	uv run mlt train configs/iris.yaml

demo: train  ## train then run one inference
	uv run mlt infer iris-rf --features 6.3,3.3,6.0,2.5

clean:
	rm -rf artifacts mlruns mlartifacts mlflow.db .pytest_cache .ruff_cache
