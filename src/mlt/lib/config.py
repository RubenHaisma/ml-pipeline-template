"""Typed config loading. One YAML file in, one validated dict out.

Kept deliberately small: a dataclass per config shape, a loader that fails
loud with a :class:`CliError` instead of a stack trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mlt.output import CliError


@dataclass(slots=True)
class TrainConfig:
    """Everything a training run needs, resolved from YAML + defaults."""

    name: str
    dataset: str = "iris"
    model: str = "random_forest"
    params: dict[str, Any] = field(default_factory=dict)
    test_size: float = 0.2
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        p = Path(path)
        if not p.is_file():
            raise CliError(f"config not found: {p}")
        try:
            raw = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - exercised via CLI
            raise CliError(f"invalid yaml in {p}: {exc}") from exc
        if "name" not in raw:
            raise CliError(f"config {p} is missing required key: name")
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})
