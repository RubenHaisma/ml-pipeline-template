"""The reference pipeline: a tiny tabular classifier on the iris dataset.

Deliberately trivial *as ML* — the point of the template is the operational
shell around it (CLI, tracking, eval contract, serving), which the domain
repos (RL, vision, classic) replace with something real. Everything here
runs on CPU in under a second so the happy path works on any laptop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from mlt.lib.config import TrainConfig
from mlt.output import CliError

_MODELS = {
    "random_forest": RandomForestClassifier,
    "logistic": LogisticRegression,
}


def _split(cfg: TrainConfig) -> tuple[Any, Any, Any, Any]:
    if cfg.dataset != "iris":
        raise CliError(f"template only ships the 'iris' dataset, got: {cfg.dataset}")
    data = load_iris()
    return train_test_split(data.data, data.target, test_size=cfg.test_size, random_state=cfg.seed)


def train(cfg: TrainConfig, out_dir: Path) -> dict[str, Any]:
    """Train, evaluate against an honest baseline, persist the model.

    Returns a result dict suitable for both ``--json`` output and MLflow.
    Reporting the baseline alongside the model is non-negotiable house style:
    a metric without a baseline is marketing, not evaluation.
    """
    if cfg.model not in _MODELS:
        raise CliError(f"unknown model '{cfg.model}', choose from {sorted(_MODELS)}")

    x_tr, x_te, y_tr, y_te = _split(cfg)

    model = _MODELS[cfg.model](**cfg.params)
    model.fit(x_tr, y_tr)
    pred = model.predict(x_te)

    baseline = DummyClassifier(strategy="most_frequent").fit(x_tr, y_tr)
    base_pred = baseline.predict(x_te)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)

    acc = float(accuracy_score(y_te, pred))
    base_acc = float(accuracy_score(y_te, base_pred))
    return {
        "name": cfg.name,
        "model": cfg.model,
        "model_path": str(model_path),
        "metrics": {
            "accuracy": round(acc, 4),
            "f1_macro": round(float(f1_score(y_te, pred, average="macro")), 4),
            "baseline_accuracy": round(base_acc, 4),
            "lift_over_baseline": round(acc - base_acc, 4),
        },
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }


def predict(model_path: Path, features: list[float]) -> dict[str, Any]:
    if not model_path.is_file():
        raise CliError(f"model not found: {model_path} (train one first)")
    model = joblib.load(model_path)
    x = np.asarray(features, dtype=float).reshape(1, -1)
    label = int(model.predict(x)[0])
    out: dict[str, Any] = {"label": label, "class_name": load_iris().target_names[label]}
    if hasattr(model, "predict_proba"):
        out["proba"] = [round(float(p), 4) for p in model.predict_proba(x)[0]]
    return out
