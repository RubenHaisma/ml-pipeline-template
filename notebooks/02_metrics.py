"""Eval view — pull the latest runs from MLflow and compare against baseline.

Run with: marimo edit notebooks/02_metrics.py
"""

import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(mo):
    mo.md(
        """
        # Run Comparison
        Reads runs logged by `mlt train` and shows accuracy vs the most-frequent
        baseline. A model that does not beat its baseline is a finding, not a failure.
        """
    )
    return


@app.cell
def __():
    import os

    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    runs = mlflow.search_runs(search_all_experiments=True)
    return mlflow, runs


@app.cell
def __(mo, runs):
    cols = [c for c in runs.columns if c.startswith("metrics.") or c == "run_id"]
    view = runs[cols] if not runs.empty else runs
    mo.md("_No runs yet — run `mlt train configs/iris.yaml` first._") if runs.empty else mo.ui.table(
        view, pagination=True
    )
    return


if __name__ == "__main__":
    app.run()
