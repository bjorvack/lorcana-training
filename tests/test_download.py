"""Network-dependent smoke tests for the release downloaders.

These hit the live GitHub API, so they're skipped unless opted in
(``RUN_NETWORK_TESTS=1``). CI enables them on the main branch
job; PR jobs run the fast offline suite.
"""

from __future__ import annotations

import os

import pytest

from lorcana_training.cards.download import download_cards
from lorcana_training.config import load_config
from lorcana_training.dataset.download import download_tournaments


_RUN_NETWORK = os.environ.get("RUN_NETWORK_TESTS") == "1"
_skip = pytest.mark.skipif(not _RUN_NETWORK, reason="set RUN_NETWORK_TESTS=1 to enable")


@_skip
def test_download_cards_pinned() -> None:
    cfg = load_config()
    path, cards = download_cards(cfg.scraper_repo, cfg.cards_release_tag)
    assert path.exists()
    assert len(cards.cards) > 1000  # real set of ~2900 printings
    # re-downloading is a no-op thanks to the sha256-verified cache
    path2, _ = download_cards(cfg.scraper_repo, cfg.cards_release_tag)
    assert path == path2


@_skip
def test_download_tournaments_pinned() -> None:
    cfg = load_config()
    path, dataset = download_tournaments(cfg.scraper_repo, cfg.tournaments_release_tag)
    assert path.exists()
    assert dataset.cards_release_tag == cfg.cards_release_tag, (
        "tournaments release is pinned to a different cards-vN than our config; "
        "bump one of them in lock-step"
    )
    assert len(dataset.tournaments) > 0
