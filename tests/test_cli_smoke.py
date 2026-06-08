"""Smoke tests — the contract CI enforces on every push.

These are deliberately about the *shell*, not ML quality: does the CLI run,
is ``--json`` valid JSON, are exit codes load-bearing, does the full
train -> infer loop work end to end on CPU.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from mlt.cli import app

runner = CliRunner()


def test_version_json():
    result = runner.invoke(app, ["version", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["version"]


def test_doctor_json_is_valid_json():
    result = runner.invoke(app, ["doctor", "--json"])
    payload = json.loads(result.stdout)
    assert "ok" in payload and "checks" in payload


def test_train_then_infer(tmp_path):
    out = str(tmp_path / "artifacts")
    trained = runner.invoke(app, ["train", "configs/iris.yaml", "--out", out, "--json"])
    assert trained.exit_code == 0, trained.stdout
    payload = json.loads(trained.stdout)
    assert payload["ok"] is True
    # the model must beat the most-frequent baseline
    assert payload["metrics"]["lift_over_baseline"] > 0

    inferred = runner.invoke(
        app, ["infer", "iris-rf", "--out", out, "--features", "6.3,3.3,6.0,2.5", "--json"]
    )
    assert inferred.exit_code == 0, inferred.stdout
    assert json.loads(inferred.stdout)["ok"] is True


def test_bad_config_exits_nonzero():
    result = runner.invoke(app, ["train", "does-not-exist.yaml", "--json"])
    assert result.exit_code != 0
    assert json.loads(result.stdout)["ok"] is False
