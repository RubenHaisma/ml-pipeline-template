"""``mlt infer`` — local smoke against a trained model. The serving stand-in."""

from __future__ import annotations

from pathlib import Path

import typer

from mlt.lib import pipeline
from mlt.output import CliError, emit, fail


def infer(
    name: str = typer.Argument(..., help="project name (artifacts/<name>/model.joblib)"),
    features: str = typer.Option(
        "5.1,3.5,1.4,0.2", "--features", help="comma-separated feature vector"
    ),
    out: str = typer.Option("artifacts", "--out", help="artifacts root"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    try:
        vec = [float(x) for x in features.split(",") if x.strip()]
        model_path = Path(out) / name / "model.joblib"
        result = pipeline.predict(model_path, vec)
    except ValueError:
        fail(
            CliError(f"--features must be comma-separated numbers, got: {features!r}"),
            json_out=json_out,
        )
        return
    except CliError as exc:
        fail(exc, json_out=json_out)
        return

    human = f"[green]{result['class_name']}[/green] (label {result['label']})"
    emit({"ok": True, **result}, json_out=json_out, human=human)
