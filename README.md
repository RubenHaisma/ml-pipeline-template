# ml-pipeline-template

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

```bash
uv sync --extra dev          # install
uv run mlt doctor            # environment readiness check (--json for CI)
uv run mlt train configs/iris.yaml
uv run mlt infer iris-rf --features 6.3,3.3,6.0,2.5
```

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

## License

Apache-2.0.
