# AGENTS.md

Agent instructions for **ml-pipeline-template**, in the cross-tool [AGENTS.md](https://agents.md) format — read natively by Codex, Cursor, GitHub Copilot, Windsurf, Amp, Devin, and others. Claude Code reads `CLAUDE.md`, which is a **symlink to this file**, so there is a single source of truth for every tool. Humans: see [README.md](README.md).

## What this repo is

A CLI-first ML pipeline template (`mlt`): train → eval-against-baseline → serve, tracked in MLflow. The ML is trivial on purpose (a tabular classifier); the value is the operational shell, which the domain repos (RL, vision, classic ML) reuse.

## Driving the CLI as an agent

Every command is **non-interactive**, takes **`--json`**, and uses **load-bearing exit codes** — so any agent (or script) can drive the whole loop and parse results without screen-scraping, a TTY, or a running service.

```bash
mlt doctor --json
# -> {"ok": true, "checks": {...}}                       exit 0 when ready, non-zero otherwise

mlt train configs/iris.yaml --json
# -> {"ok": true, "name": "iris-rf", "metrics": {"accuracy": 1.0, "lift_over_baseline": 0.7, ...}}

mlt infer iris-rf --features 6.3,3.3,6.0,2.5 --json
# -> {"ok": true, "label": 2, "class_name": "virginica", "proba": [...]}
```

**Contract:** with `--json`, stdout is exactly one JSON object (success *or* failure: `{"ok": false, "error": "..."}`). Exit `0` = success, non-zero = failure with one line on stderr. So: parse stdout, branch on the exit code. Discover the surface with `mlt --help` and `mlt <cmd> --help`.

## Setup (for the agent's environment)

```bash
uv sync --extra dev      # install (CPU, fast)
uv run mlt doctor --json # confirm the environment is ready
```

## Hard rules (when editing this repo)

1. **CLI-first.** Every capability ships as an `mlt` subcommand. No notebook-only or UI-only flows.
2. **`--json` on every command.** Machine-readable output is a contract; don't add a command without it.
3. **Exit codes mean something.** `0` success; non-zero failure with one human-readable line on stderr (see `src/mlt/output.py`). Never swallow errors.
4. **MLflow is the single source of truth** for runs/params/metrics. No ad-hoc dotfiles. Falls back to `sqlite:///mlflow.db` with no server.
5. **Report a baseline with every metric.** A model that doesn't beat its baseline is a finding to surface, not a number to hide.
6. **Marimo `.py`, never `.ipynb`.** JSON notebooks don't diff, grep, or edit surgically.

## Build / test / verify

```bash
uv run ruff check src tests scripts     # lint
uv run pytest                           # smoke suite (runs the real train→infer loop)
make repro                              # determinism: train twice, assert identical metrics
make readme                             # run the README's <!-- ci-test --> commands
```

## Layout

```
src/mlt/
  cli.py        # Typer app, one command per capability
  output.py     # emit()/fail() — the output + exit-code contract
  commands/     # one file per subcommand
  lib/          # config, tracking, pipeline (in-process, no shellouts)
notebooks/      # marimo .py
configs/        # yaml, one per run shape
scripts/        # stdlib CI helpers (ci_report, test_readme, check_repro)
```

## What NOT to add

- `.ipynb` notebooks. Cloud-specific SDKs (SageMaker/Vertex/Azure ML). A web dashboard (use MLflow's). Commands without `--json`.
