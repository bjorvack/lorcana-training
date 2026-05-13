"""Typed access to ``config/training.yaml``.

The pipeline has a single config file. Everything else (CLI flags,
env vars, ...) derives from it so there is never ambiguity about
which artifact a run was trained against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "training.yaml"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    raw: dict[str, Any]

    @property
    def schemas_repo(self) -> str:
        return cast(str, self.raw["schemas_repo"])

    @property
    def schemas_release_tag(self) -> str:
        return cast(str, self.raw["schemas_release_tag"])

    @property
    def scraper_repo(self) -> str:
        return cast(str, self.raw["scraper_repo"])

    @property
    def cards_release_tag(self) -> str:
        return cast(str, self.raw["cards_release_tag"])

    @property
    def tournaments_release_tag(self) -> str:
        return cast(str, self.raw["tournaments_release_tag"])

    @property
    def encoder_release_tag(self) -> str | None:
        val = self.raw.get("encoder_release_tag")
        return cast(str, val) if val else None


def load_config(path: Path | str | None = None) -> TrainingConfig:
    """Load ``config/training.yaml`` (or the path given)."""
    resolved = Path(path) if path else DEFAULT_CONFIG_PATH
    with resolved.open("r", encoding="utf8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{resolved}: expected a YAML mapping, got {type(data).__name__}")
    return TrainingConfig(raw=data)
