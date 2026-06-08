"""``mlt train <config>`` — train, evaluate vs baseline, log everything to MLflow."""

from __future__ import annotations

from pathlib import Path

import typer

from mlt.lib import pipeline, tracking
from mlt.lib.config import TrainConfig
from mlt.output import CliError, emit, fail


def train(
    config: str = typer.Argument(..., help="path to a train config yaml"),
    out: str = typer.Option("artifacts", "--out", help="where to write the model"),
    json_out: bool = typer.Option(False, "--json", help="machine-readable output"),
) -> None:
    try:
        cfg = TrainConfig.from_yaml(config)
        out_dir = Path(out) / cfg.name
        result = pipeline.train(cfg, out_dir)

        with tracking.run(experiment=cfg.name, run_name=cfg.model):
            tracking.log_params({"model": cfg.model, "seed": cfg.seed, **cfg.params})
            tracking.log_metrics(result["metrics"])
    except CliError as exc:
        fail(exc, json_out=json_out)
        return

    m = result["metrics"]
    human = (
        f"[green]trained[/green] {cfg.name} ({cfg.model})\n"
        f"  accuracy {m['accuracy']}  (baseline {m['baseline_accuracy']}, "
        f"lift {m['lift_over_baseline']:+})\n"
        f"  model -> {result['model_path']}"
    )
    emit({"ok": True, **result}, json_out=json_out, human=human)
