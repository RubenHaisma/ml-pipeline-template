.PHONY: install up down fmt lint test doctor train demo repro readme summary notebooks clean

install:  ## uv sync with dev extras
	uv sync --extra dev

up:  ## bring up MLflow on :5050
	docker compose up -d

down:  ## stop services (volumes preserved)
	docker compose down

fmt:  ## ruff format
	uv run ruff format src tests scripts

lint:  ## ruff check
	uv run ruff check src tests scripts

test:  ## pytest smoke suite
	uv run pytest

doctor:  ## environment readiness check
	uv run mlt doctor

train:  ## train the reference config
	uv run mlt train configs/iris.yaml

demo: train  ## train then run one inference
	uv run mlt infer iris-rf --features 6.3,3.3,6.0,2.5

repro:  ## prove training is deterministic (same seed → identical metrics)
	uv run python scripts/check_repro.py -- uv run mlt train configs/iris.yaml

readme:  ## run the README's ci-test commands so the docs can't go stale
	uv run python scripts/test_readme.py

summary:  ## train + print the markdown metrics summary CI posts to the run page
	uv run mlt train configs/iris.yaml --json | uv run python scripts/ci_report.py

notebooks:  ## execute every marimo notebook headless (fails on a dead cell)
	@for nb in notebooks/*.py; do echo "running $$nb"; uv run python "$$nb" || exit 1; done

clean:
	rm -rf artifacts mlruns mlartifacts mlflow.db metrics.json .pytest_cache .ruff_cache
