"""Fetch a pinned ``tournaments-vN`` release from ``lorcana-scraper``."""

from __future__ import annotations

import json
from pathlib import Path

from ..release.download import download_asset
from ..schemas.generated.dataset import Dataset


def download_tournaments(repo: str, tag: str) -> tuple[Path, Dataset]:
    """Download and parse ``dataset.json`` from ``<repo>@<tag>``."""
    sha_path = download_asset(repo, tag, "dataset.json.sha256")
    expected = sha_path.read_text(encoding="utf8").strip().split()[0]
    dataset_path = download_asset(repo, tag, "dataset.json", expected_sha256=expected)
    data = json.loads(dataset_path.read_text(encoding="utf8"))
    return dataset_path, Dataset.model_validate(data)
