"""Unit tests for the ``(name, version)`` collapse and vocab build.

Uses small hand-built ``CardSet`` fixtures so the behaviour is
explicit. A separate opt-in network test runs the full pipeline
against the pinned ``cards-vN`` to catch regressions on real data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lorcana_training.cards.logical import build_logical_cards, logical_id
from lorcana_training.cards.vocab import build_vocab, write_vocab
from lorcana_training.schemas.generated.card_set import CardSet


def _card(
    id_: str,
    name: str,
    version: str | None,
    set_code: str,
    number: int,
    text: str = "",
    cost: int = 3,
    inks: tuple[str, ...] = ("Amber",),
    types: tuple[str, ...] = ("Character",),
) -> dict:
    return {
        "id": id_,
        "name": name,
        "version": version,
        "setCode": set_code,
        "cardNumber": number,
        "cost": cost,
        "inkwell": True,
        "inks": list(inks),
        "types": list(types),
        "classifications": ["Storyborn"] if "Character" in types else [],
        "keywords": [],
        "text": text,
        "flavor": None,
        "imageUrl": f"https://example.test/{id_}.avif",
        "legality": "legal",
        "lore": 2,
        "strength": 3,
        "willpower": 3,
        "moveCost": None,
    }


def _card_set(*cards: dict) -> CardSet:
    return CardSet.model_validate(
        {
            "cardSetVersion": "sha256:test",
            "fetchedAt": "2026-05-13T00:00:00Z",
            "cards": list(cards),
        }
    )


def test_collapses_same_name_and_version_across_sets() -> None:
    cs = _card_set(
        _card("crd_a", "Mickey", "True Friend", "1", 12),
        _card("crd_b", "Mickey", "True Friend", "P3", 10),  # promo reprint
        _card("crd_c", "Mickey", "True Friend", "10", 500),  # enchanted in set 10
        _card("crd_d", "Mickey", "Brave Little Tailor", "1", 115),  # different version
    )
    logical = build_logical_cards(cs)
    assert len(logical.cards) == 2  # True Friend + Brave Little Tailor
    assert logical.report.total_printings == 4
    assert logical.report.groups_with_multiple_printings == 1


def test_canonical_prefers_highest_numeric_set() -> None:
    cs = _card_set(
        _card("crd_promo", "Elsa", "Snow Queen", "P1", 1),
        _card("crd_set1", "Elsa", "Snow Queen", "1", 42),
        _card("crd_set5", "Elsa", "Snow Queen", "5", 42),
        _card("crd_set10", "Elsa", "Snow Queen", "10", 42),  # newest
    )
    logical = build_logical_cards(cs)
    assert len(logical.cards) == 1
    assert logical.cards[0].canonical.id == "crd_set10"
    # All 4 printings map to the same logical id.
    assert set(logical.printing_to_logical_id.values()) == {logical.cards[0].logical_id}
    assert len(logical.cards[0].printings) == 4


def test_canonical_falls_back_to_alpha_set_if_no_numeric() -> None:
    cs = _card_set(
        _card("crd_promo1", "Only Promo", "Card", "P1", 1),
        _card("crd_promo2", "Only Promo", "Card", "P3", 1),
    )
    logical = build_logical_cards(cs)
    # Deterministic but which wins doesn't matter much; we just want
    # a stable canonical. `_setcode_rank` with `reverse=True` picks the
    # lexicographically higher non-numeric code.
    assert logical.cards[0].canonical.id == "crd_promo2"


def test_actions_without_version_collapse_correctly() -> None:
    cs = _card_set(
        _card("crd_dragon_fire", "Dragon Fire", None, "1", 95, types=("Action",)),
        _card("crd_dragon_fire2", "Dragon Fire", None, "P1", 20, types=("Action",)),
    )
    logical = build_logical_cards(cs)
    assert len(logical.cards) == 1
    assert logical.cards[0].version == ""
    assert logical.cards[0].logical_id == logical_id(("Dragon Fire", ""))


def test_functional_drift_is_reported_but_does_not_fail() -> None:
    cs = _card_set(
        _card("crd_x", "Ursula", "Deceiver", "3", 90, text="this character"),
        _card("crd_y", "Ursula", "Deceiver", "D23", 3, text="this card"),  # text differs
    )
    logical = build_logical_cards(cs)
    assert len(logical.cards) == 1
    assert logical.report.groups_with_functional_drift == 1
    # Logged as a sample for human review.
    assert logical.report.drift_samples[0]["logical_id"] == "Ursula|Deceiver"


def test_build_vocab_and_write(tmp_path: Path) -> None:
    cs = _card_set(
        _card("crd_a", "Mickey", "True Friend", "1", 12),
        _card("crd_b", "Mickey", "True Friend", "10", 500),
        _card("crd_c", "Elsa", "Snow Queen", "1", 42),
    )
    logical = build_logical_cards(cs)
    vocab = build_vocab(logical)
    # PAD + 2 logical cards.
    assert vocab.size == 2
    assert vocab.pad_index == 0
    assert vocab.entries[0] is None
    idx_map = vocab.build_index_map()
    assert idx_map == {"Elsa|Snow Queen": 1, "Mickey|True Friend": 2}  # sorted alphabetically

    paths = write_vocab(
        vocab,
        logical,
        cs,
        out_dir=tmp_path,
        cards_release_tag="cards-vtest",
    )
    vdoc = json.loads(paths["vocab"].read_text())
    assert vdoc["padIndex"] == 0
    assert vdoc["size"] == 2
    assert vdoc["cardsReleaseTag"] == "cards-vtest"
    # canonical printing is the highest-numeric set printing
    mickey = next(c for c in vdoc["cards"] if c["logicalId"] == "Mickey|True Friend")
    assert mickey["canonicalPrintingId"] == "crd_b"
    assert set(mickey["printingIds"]) == {"crd_a", "crd_b"}

    ptl = json.loads(paths["printing_to_logical"].read_text())
    assert ptl == {
        "crd_a": "Mickey|True Friend",
        "crd_b": "Mickey|True Friend",
        "crd_c": "Elsa|Snow Queen",
    }


@pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1", reason="set RUN_NETWORK_TESTS=1 to enable"
)
def test_real_cards_collapse_ratio() -> None:
    # Pinned cards-v2026.05.13-01: 2911 printings → 2282 logical.
    from lorcana_training.cards.download import download_cards
    from lorcana_training.config import load_config

    cfg = load_config()
    _, cs = download_cards(cfg.scraper_repo, cfg.cards_release_tag)
    logical = build_logical_cards(cs)
    # Be loose: exact numbers will drift as new cards ship. The ratio
    # is the property we care about: some collapse happens.
    assert logical.report.total_printings >= len(logical.cards) + 100
    assert logical.report.groups_with_multiple_printings >= 100
