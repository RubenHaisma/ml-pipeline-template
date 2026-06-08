# Architecture

```
┌────────────────────────────────────────────────┐
│  mlt CLI (Typer)                                 │
│  doctor · train · infer · version                │
└───────┬───────────────────────────────┬─────────┘
        │                                │
        ▼                                ▼
  ┌───────────┐                    ┌───────────┐
  │ lib/      │  ── trains ──►     │  MLflow   │
  │ pipeline  │                    │  runs +   │
  │ (sklearn) │  ── logs ────────► │  metrics  │
  └───────────┘                    └───────────┘
        │
        ▼
  artifacts/<name>/model.joblib  ──►  mlt infer
```

## Output contract (`output.py`)
Every command funnels through `emit()` (success) or `fail()` (error). This is
what makes `--json` and exit codes uniform instead of per-command guesswork:

- `--json` → exactly one JSON object on stdout, success or failure.
- human mode → rich-formatted line(s) on stdout, `error: ...` on stderr.
- failure → non-zero exit, always.

## Tracking (`lib/tracking.py`)
MLflow with a local `sqlite:///mlflow.db` fallback so a fresh checkout works
with no services (MLflow 3 deprecated the bare file store; sqlite is the
supported local backend). Export `MLFLOW_TRACKING_URI=http://localhost:5050`
(after `make up`) to use the shared backend in `docker-compose.yml`.

## Why this shape
The ML here is trivial on purpose. The reusable value is the shell: a CLI an
agent can drive end-to-end, machine-readable everywhere, with tracking and an
honest eval baked in. Each domain repo swaps `lib/pipeline.py` for something
real (GRPO, defect detection, forecasting) and keeps everything else.
