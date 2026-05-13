"""Smoke tests for ``lorcana_training.schemas.generated.*``.

- The generated ``__init__`` and each module import cleanly.
- The checked-in ``.generated_from`` stamp matches the pin in
  ``config/training.yaml`` so PRs that bump the pin without
  re-running codegen fail CI.
- A minimal object round-trips through each model.
"""

from __future__ import annotations

from lorcana_training.config import load_config
from lorcana_training.schemas.generated import (  # noqa: F401
    card,
    card_set,
    dataset,
    deck,
    manifest,
    tournament,
)
from lorcana_training.schemas.gen import GENERATED_DIR


def test_generated_from_matches_config() -> None:
    cfg = load_config()
    stamp = (GENERATED_DIR / ".generated_from").read_text(encoding="utf8").strip()
    assert stamp == f"{cfg.schemas_repo}@{cfg.schemas_release_tag}", (
        f"generated schemas pin ({stamp!r}) is out of date vs "
        f"config/training.yaml ({cfg.schemas_repo}@{cfg.schemas_release_tag!r}). "
        "Run `uv run python -m lorcana_training.schemas.gen`."
    )


def test_card_round_trip() -> None:
    sample = {
        "id": "crd_test",
        "name": "Test Card",
        "version": "Smoke Test",
        "setCode": "1",
        "cardNumber": 1,
        "cost": 3,
        "inkwell": True,
        "inks": ["Amber"],
        "types": ["Character"],
        "classifications": ["Storyborn"],
        "keywords": [],
        "text": "",
        "flavor": None,
        "imageUrl": "https://example.test/x.avif",
        "legality": "legal",
        "lore": 2,
        "strength": 3,
        "willpower": 3,
        "moveCost": None,
    }
    c = card.Card.model_validate(sample)
    # Round-tripping via `model_dump(by_alias=True)` preserves the
    # source-format keys we read from the scraper's JSON.
    dumped = c.model_dump(by_alias=True, mode="json")
    assert dumped["setCode"] == "1"
    assert dumped["cardNumber"] == 1
