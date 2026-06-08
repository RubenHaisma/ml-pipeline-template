# ml-pipeline-template — agent rules

For Claude Code and similar agents. Humans read `README.md`.

## What this repo is
The house style for a set of production ML reference repos. A CLI-first
pipeline (`mlt`) around a trivial tabular model, where the *operational shell*
is the point: CLI, MLflow tracking, eval-with-baseline, serving smoke. Domain
repos (RL, vision, classic ML) replace the model, keep the shell.

## Hard rules
1. **CLI-first.** Every capability ships as an `mlt` subcommand. No notebook-
   only or UI-only flows.
2. **`--json` on every command.** Machine-readable output is a contract, not a
   nicety. Don't add a command without it.
3. **Exit codes mean something.** `0` success; non-zero failure with one
   human-readable line on stderr (see `output.py`). Never swallow errors.
4. **MLflow is the single source of truth** for runs/params/metrics. No state
   in ad-hoc dotfiles. Falls back to `file:./mlruns` with no server running.
5. **Report a baseline with every metric.** A model that doesn't beat its
   baseline is a finding to surface, not a number to hide.
6. **Marimo `.py`, never `.ipynb`.** JSON notebooks don't diff, grep, or edit
   surgically.

## Layout
```
src/mlt/
  cli.py            # Typer app, one command per capability
  output.py         # emit()/fail() — the output + exit-code contract
  commands/         # one file per subcommand
  lib/              # config, tracking, pipeline (in-process, no shellouts)
notebooks/          # marimo .py
configs/            # yaml, one per run shape
tests/              # pytest smoke — runs the real train->infer loop on CPU
```

## What NOT to add
- `.ipynb` notebooks.
- Cloud-specific SDKs (SageMaker/Vertex/Azure ML) — stays local + compose.
- A web dashboard — use MLflow's UI.
- Commands without `--json`.
