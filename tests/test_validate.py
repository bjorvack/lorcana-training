"""Tests for dataset validation + printing->logical rewrite.

Covers the two pieces together because legality is only meaningful
after the rewrite has collapsed printings to logical cards.
"""

from __future__ import annotations

from lorcana_training.cards.logical import build_logical_cards
from lorcana_training.cards.vocab import build_vocab
from lorcana_training.dataset.rewrite import rewrite_deck
from lorcana_training.dataset.validate import validate_dataset
from lorcana_training.schemas.generated.card_set import CardSet
from lorcana_training.schemas.generated.dataset import Dataset


def _card(
    id_: str,
    name: str,
    *,
    version: str | None = None,
    set_code: str = "1",
    number: int = 1,
    cost: int = 3,
    inks: tuple[str, ...] = ("Amber",),
    types: tuple[str, ...] = ("Character",),
    text: str = "",
    legality: str = "legal",
    inkwell: bool = True,
) -> dict:
    return {
        "id": id_,
        "name": name,
        "version": version,
        "setCode": set_code,
        "cardNumber": number,
        "cost": cost,
        "inkwell": inkwell,
        "inks": list(inks),
        "types": list(types),
        "classifications": ["Storyborn"] if "Character" in types else [],
        "keywords": [],
        "text": text,
        "flavor": None,
        "imageUrl": f"https://example.test/{id_}.avif",
        "legality": legality,
        "lore": 2,
        "strength": 3,
        "willpower": 3,
        "moveCost": None,
    }


def _cs(*cards: dict) -> CardSet:
    return CardSet.model_validate(
        {
            "cardSetVersion": "sha256:test",
            "fetchedAt": "2026-05-13T00:00:00Z",
            "cards": list(cards),
        }
    )


def _deck(cards: list[tuple[str, int]], inks: tuple[str, ...] = ("Amber",)) -> dict:
    return {
        "inks": list(inks),
        "cards": [{"cardId": cid, "count": n} for cid, n in cards],
        "name": None,
        "source": "test",
    }


def _dataset(*decks: dict, cards_release_tag: str = "cards-vtest") -> Dataset:
    return Dataset.model_validate(
        {
            "datasetVersion": "0.1.0",
            "schemaVersion": "0.4.0",
            "cardSetVersion": "sha256:test",
            "cardsReleaseTag": cards_release_tag,
            "generatedAt": "2026-05-13T00:00:00Z",
            "sources": ["test.local"],
            "tournaments": [
                {
                    "sourceUrl": "https://example.test/t1",
                    "sourceName": "test.local",
                    "name": "Test Tournament",
                    "date": "2025-06-01",
                    "decks": [
                        {"placement": i, "player": f"p{i}", "deck": d}
                        for i, d in enumerate(decks, start=1)
                    ],
                }
            ],
        }
    )


def _valid_60_card_deck_cards() -> list[tuple[str, int]]:
    """15 cards x 4 = 60. Uses ids ``crd_a`` .. ``crd_o``."""
    return [(f"crd_{chr(ord('a') + i)}", 4) for i in range(15)]


def test_rewrite_collapses_printings_of_same_logical_card() -> None:
    # Two printings of Mickey collapse into one logical index with count
    # summed across them.
    cs = _cs(
        _card("crd_base", "Mickey", version="True Friend", number=12),
        _card("crd_ench", "Mickey", version="True Friend", set_code="10", number=500),
    )
    logical = build_logical_cards(cs)
    vocab = build_vocab(logical)
    idx_by_id = vocab.build_index_map()

    # Build a dataset with a single deck and use the validate path to get
    # at the parsed (inner) deck model without having to reach into the
    # generated dataset.Deck1 by name.
    ds = _dataset(_deck([("crd_base", 2), ("crd_ench", 3)]))
    inner = ds.tournaments[0].decks[0].deck
    result = rewrite_deck(
        inner,
        printing_to_logical_id=logical.printing_to_logical_id,
        logical_index_by_id=idx_by_id,
    )
    assert result.ok
    assert result.deck is not None
    # Single logical entry with 2+3=5 copies.
    assert len(result.deck.cards) == 1
    _, count = result.deck.cards[0]
    assert count == 5


def test_validate_drops_deck_smaller_than_60() -> None:
    cs = _cs(*(_card(f"crd_{chr(ord('a') + i)}", f"Card{i}") for i in range(15)))
    # Deck with only 40 cards; upstream unresolved cards would produce this shape.
    ds = _dataset(_deck([(f"crd_{chr(ord('a') + i)}", 4) for i in range(10)]))
    logical = build_logical_cards(cs)
    vocab = build_vocab(logical)
    vd = validate_dataset(ds, vocab=vocab, logical_cards=logical)
    assert vd.report.valid_decks == 0
    assert vd.report.total_decks == 1
    reasons = dict(vd.report.drop_reasons_histogram)
    assert reasons.get("deck_size_below_60") == 1


def test_validate_accepts_clean_60_card_deck() -> None:
    cs = _cs(*(_card(f"crd_{chr(ord('a') + i)}", f"Card{i}") for i in range(15)))
    ds = _dataset(_deck(_valid_60_card_deck_cards()))
    logical = build_logical_cards(cs)
    vocab = build_vocab(logical)
    vd = validate_dataset(ds, vocab=vocab, logical_cards=logical)
    assert vd.report.valid_decks == 1
    assert vd.report.drop_rate == 0.0
    accepted = vd.decks[0]
    assert accepted.deck.total_cards == 60
    assert accepted.tournament_name == "Test Tournament"


def test_validate_drops_banned_cards_always() -> None:
    cs = _cs(
        *(_card(f"crd_{chr(ord('a') + i)}", f"Card{i}") for i in range(14)),
        _card("crd_o", "BannedCard", legality="banned"),
    )
    ds = _dataset(_deck(_valid_60_card_deck_cards()))
    logical = build_logical_cards(cs)
    vocab = build_vocab(logical)
    vd = validate_dataset(ds, vocab=vocab, logical_cards=logical)
    assert vd.report.valid_decks == 0
    reasons = dict(vd.report.drop_reasons_histogram)
    assert reasons.get("card_not_legal") == 1


def test_validate_keeps_not_legal_by_default() -> None:
    """`not_legal` means 'rotated out of current format' — most
    historic tournament decks contain these. Default mode keeps them."""
    cs = _cs(
        *(_card(f"crd_{chr(ord('a') + i)}", f"Card{i}") for i in range(14)),
        _card("crd_o", "RotatedCard", legality="not_legal"),
    )
    ds = _dataset(_deck(_valid_60_card_deck_cards()))
    logical = build_logical_cards(cs)
    vocab = build_vocab(logical)
    vd = validate_dataset(ds, vocab=vocab, logical_cards=logical)
    assert vd.report.valid_decks == 1


def test_validate_strict_legality_drops_not_legal() -> None:
    cs = _cs(
        *(_card(f"crd_{chr(ord('a') + i)}", f"Card{i}") for i in range(14)),
        _card("crd_o", "RotatedCard", legality="not_legal"),
    )
    ds = _dataset(_deck(_valid_60_card_deck_cards()))
    logical = build_logical_cards(cs)
    vocab = build_vocab(logical)
    vd = validate_dataset(ds, vocab=vocab, logical_cards=logical, strict_legality=True)
    assert vd.report.valid_decks == 0


def test_validate_flags_max_copies_violation_after_collapse() -> None:
    """Two printings of the same logical card summing to >4 copies is
    the whole point of the logical collapse — verify it trips."""
    cs = _cs(
        _card("crd_base", "Mickey", version="True Friend", number=12),
        _card("crd_ench", "Mickey", version="True Friend", set_code="10", number=500),
        *(_card(f"crd_{chr(ord('a') + i)}", f"Filler{i}") for i in range(14)),
    )
    # 4x base + 4x enchanted = 8 copies of one logical Mickey, rest filler to 60.
    cards = [("crd_base", 4), ("crd_ench", 4)] + [(f"crd_{chr(ord('a') + i)}", 4) for i in range(13)]
    ds = _dataset(_deck(cards))
    logical = build_logical_cards(cs)
    vocab = build_vocab(logical)
    vd = validate_dataset(ds, vocab=vocab, logical_cards=logical)
    assert vd.report.valid_decks == 0
    reasons = dict(vd.report.drop_reasons_histogram)
    assert reasons.get("count_exceeds_max_copies") == 1
