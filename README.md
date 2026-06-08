# ml-pipeline-template

[![ci](https://github.com/RubenHaisma/ml-pipeline-template/actions/workflows/ci.yml/badge.svg)](https://github.com/RubenHaisma/ml-pipeline-template/actions/workflows/ci.yml)

**CLI-first ML pipeline template.** Train → eval-against-baseline → serve, tracked in MLflow, driven by one machine-readable binary. The house style for a set of production ML reference repos — [`rl-studio`](https://github.com/rubenhaisma/rl-studio), [`vision-pipeline`](https://github.com/rubenhaisma/vision-pipeline), and [`ml-pipeline`](https://github.com/rubenhaisma/ml-pipeline) all derive from it.

> The ML here is deliberately trivial (a tabular classifier on iris). The point is the **operational shell**: a CLI an agent or a human can drive end-to-end, `--json` everywhere, load-bearing exit codes, MLflow as the single source of truth, and an honest baseline reported with every metric. The domain repos swap the model; they keep the shell.

## Why it looks like this

Most ML demos are a notebook that works once on the author's laptop. This is the opposite: a script-not-a-ritual pipeline that runs the same way in CI, in an agent loop, and on a fresh checkout.

- **CLI-first** — every capability is an `mlt` subcommand. No notebook-only happy paths.
- **`--json` on every command** — output is a contract, so tools and agents compose it.
- **Exit codes mean something** — `0` ok, non-zero failure with one line on stderr.
- **MLflow is the source of truth** — falls back to a local `file:./mlruns` store with zero services running.
- **Baseline with every metric** — a model that doesn't beat its baseline is a finding, not a number to hide.
- **Marimo `.py`, never `.ipynb`** — notebooks that diff, grep, and edit like source.

## Quickstart

<!-- ci-test -->
```bash
uv sync --extra dev          # install
uv run mlt doctor            # environment readiness check (--json for CI)
uv run mlt train configs/iris.yaml
uv run mlt infer iris-rf --features 6.3,3.3,6.0,2.5
```

> The block above is marked `<!-- ci-test -->` — **CI runs these exact commands on every push**, so this quickstart can never silently drift from the code.

Output of `train` (human mode):

```
trained iris-rf (random_forest)
  accuracy 0.9667  (baseline 0.3, lift +0.6667)
  model -> artifacts/iris-rf/model.joblib
```

Everything is also available via `make`: `make demo` runs the full train→infer loop.

## CLI surface

```
mlt doctor [--json]                 # is this environment ready?
mlt train <config> [--out] [--json] # train, eval vs baseline, log to MLflow
mlt infer <name> [--features] [--json]
mlt version [--json]
```

Tracking UI (optional): `make up` starts MLflow on `localhost:5050`, then
`export MLFLOW_TRACKING_URI=http://localhost:5050`.

## Notebooks (marimo)

```bash
uv run marimo edit notebooks/01_explore.py   # feature distributions, class balance
uv run marimo edit notebooks/02_metrics.py   # compare MLflow runs vs baseline
```

## What's verified

| Path                              | Status            |
| --------------------------------- | ----------------- |
| `mlt train` / `infer` on CPU      | ✅ verified        |
| `pytest` smoke suite + ruff in CI | ✅ verified        |
| MLflow local sqlite store         | ✅ verified        |
| MLflow server via docker-compose  | 🟡 compose provided, runs locally |

## Use it as a starting point

Fork it, replace `src/mlt/lib/pipeline.py` with your domain (training,
evaluation, serving), keep the CLI / output / tracking shell. The three domain
repos linked above show exactly that, for RL fine-tuning, computer vision, and
classic ML.

## CI does more than lint

Most repos' CI checks that the code *parses*. This one checks that the *pipeline works* — three things beyond lint + tests, all stdlib, no extra deps:

1. **It runs the pipeline and publishes the numbers.** Every push trains the model and posts a live metrics table to the GitHub Actions [run summary](https://github.com/RubenHaisma/ml-pipeline-template/actions) (`scripts/ci_report.py`). The numbers in CI are produced on that commit, not pasted by hand.
2. **It keeps the docs honest.** The Quickstart block is marked `<!-- ci-test -->` and `scripts/test_readme.py` runs those exact commands in CI. Docs that drift from the code fail the build.
3. **It proves determinism.** `scripts/check_repro.py` trains twice and asserts identical metrics — a seed is a promise, and CI verifies the promise holds.

Run them locally too: `make summary`, `make readme`, `make repro`.

## License

Apache-2.0.
