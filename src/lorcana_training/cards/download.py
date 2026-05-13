"""Fetch a pinned ``cards-vN`` release from ``lorcana-scraper``.

Wrapper around :func:`lorcana_training.release.download.download_asset`
that knows the release layout (``cards.json`` + sidecar sha256) and
returns a parsed pydantic :class:`CardSet`.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..release.download import download_asset
from ..schemas.generated.card_set import CardSet


def download_cards(repo: str, tag: str) -> tuple[Path, CardSet]:
    """Download and parse ``cards.json`` from ``<repo>@<tag>``.

    The scraper publishes a ``cards.json.sha256`` sidecar alongside each
    release; we pull it first, strip any trailing whitespace, and use
    it to verify the main file. That way a corrupt cached copy cannot
    silently survive between runs.
    """
    sha_path = download_asset(repo, tag, "cards.json.sha256")
    expected = sha_path.read_text(encoding="utf8").strip().split()[0]
    cards_path = download_asset(repo, tag, "cards.json", expected_sha256=expected)
    data = json.loads(cards_path.read_text(encoding="utf8"))
    return cards_path, CardSet.model_validate(data)
