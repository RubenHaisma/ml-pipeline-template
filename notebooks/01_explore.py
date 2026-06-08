"""Dataset exploration — feature distributions and class balance.

Marimo, not Jupyter: this is plain Python that diffs, greps, and edits like
source. Run with: marimo edit notebooks/01_explore.py
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
        # Dataset Explore
        Feature distributions and class balance for the reference dataset.
        Replace `load_iris` with your domain loader in a real repo.
        """
    )
    return


@app.cell
def __():
    import pandas as pd
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    df = data.frame
    return data, df, pd


@app.cell
def __(df, mo):
    mo.ui.table(df.head(20), pagination=True)
    return


@app.cell
def __(df, mo):
    desc = df.describe().round(2)
    balance = df["target"].value_counts().to_dict()
    mo.md(f"**Class balance:** `{balance}`\n\n**Summary:**\n\n{desc.to_markdown()}")
    return


if __name__ == "__main__":
    app.run()
