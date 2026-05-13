"""Tests for the BPE card-text tokeniser."""

from __future__ import annotations

from pathlib import Path

import pytest

from lorcana_training.cards.logical import build_logical_cards
from lorcana_training.schemas.generated.card_set import CardSet
from lorcana_training.text.tokeniser import (
    CLS_TOKEN,
    GAME_GLYPHS,
    MASK_TOKEN,
    PAD_TOKEN,
    SEP_TOKEN,
    UNK_TOKEN,
    collect_reserved_tokens,
    load_tokeniser,
    special_token_ids,
    train_tokeniser,
)


_SAMPLE_TEXTS = [
    "Shift 3 (You may pay 3 {I} to play this on top of one of your characters named Elsa.)",
    "Rush (This character can challenge the turn they're played.)",
    "Evasive (Only characters with Evasive can challenge this character.)",
    "When you play this character, exert chosen opposing character.",
    "Your characters gain +1 {S} this turn.",
    "Sing Together 8 (Any number of your or your teammates' characters...)",
    "While this character has 6 {S} or more, it gets +2 {L}.",
    "When this character is banished, you may draw a card.",
    "Whenever one of your characters quests, gain 1 lore.",
]


def test_trained_tokeniser_has_expected_specials(tmp_path: Path) -> None:
    tok = train_tokeniser(_SAMPLE_TEXTS, out_path=tmp_path / "tok.json", vocab_size=500)
    ids = special_token_ids(tok)
    # The five specials are present and occupy the first slots by design.
    assert ids[PAD_TOKEN] == 0
    assert ids[UNK_TOKEN] == 1
    assert ids[CLS_TOKEN] == 2
    assert ids[SEP_TOKEN] == 3
    assert ids[MASK_TOKEN] == 4


def test_encode_wraps_in_cls_sep(tmp_path: Path) -> None:
    tok = train_tokeniser(_SAMPLE_TEXTS, out_path=tmp_path / "tok.json", vocab_size=500)
    ids = special_token_ids(tok)
    enc = tok.encode("Shift 3")
    assert enc.ids[0] == ids[CLS_TOKEN]
    assert enc.ids[-1] == ids[SEP_TOKEN]


def test_round_trip_preserves_text(tmp_path: Path) -> None:
    tok = train_tokeniser(_SAMPLE_TEXTS, out_path=tmp_path / "tok.json", vocab_size=500)
    # Strip the auto-added special tokens before decoding.
    enc = tok.encode("Rush means the character may challenge immediately.")
    ids = special_token_ids(tok)
    body = [i for i in enc.ids if i not in {ids[CLS_TOKEN], ids[SEP_TOKEN]}]
    decoded = tok.decode(body)
    # Byte-level BPE preserves spacing / case; allow leading/trailing ws.
    assert decoded.strip() == "Rush means the character may challenge immediately."


def test_unseen_text_never_produces_unk(tmp_path: Path) -> None:
    """Byte-level BPE always resolves to subword pieces, never [UNK]."""
    tok = train_tokeniser(_SAMPLE_TEXTS, out_path=tmp_path / "tok.json", vocab_size=500)
    unk_id = special_token_ids(tok)[UNK_TOKEN]
    enc = tok.encode("Resonate 5 — a mechanic not present in training.")
    assert unk_id not in enc.ids


def test_load_round_trips_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "tok.json"
    trained = train_tokeniser(_SAMPLE_TEXTS, out_path=path, vocab_size=500)
    reloaded = load_tokeniser(path)
    a = trained.encode("Shift 3")
    b = reloaded.encode("Shift 3")
    assert a.ids == b.ids


def test_empty_corpus_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        train_tokeniser([], out_path=tmp_path / "tok.json")


def _card(name: str, *, classifications: tuple[str, ...] = (), keywords: tuple[str, ...] = ()) -> dict:
    return {
        "id": f"crd_{name.lower().replace(' ', '_')}",
        "name": name,
        "version": "Test",
        "setCode": "1",
        "cardNumber": 1,
        "cost": 3,
        "inkwell": True,
        "inks": ["Amber"],
        "types": ["Character"],
        "classifications": list(classifications),
        "keywords": list(keywords),
        "text": "",
        "flavor": None,
        "imageUrl": f"https://example.test/{name}.avif",
        "legality": "legal",
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


def test_collect_reserved_tokens_includes_names_classes_keywords_glyphs() -> None:
    cs = _cs(
        _card("Elsa", classifications=("Hero", "Princess"), keywords=("Shift",)),
        _card("Tinker Bell", classifications=("Hero", "Fairy"), keywords=("Rush", "Evasive")),
    )
    logical = build_logical_cards(cs)
    reserved = collect_reserved_tokens(logical.cards)
    # Glyphs ship first so they get low, stable ids.
    for glyph in GAME_GLYPHS:
        assert glyph in reserved
    # Character short names are single atomic tokens.
    assert "Elsa" in reserved
    assert "Tinker Bell" in reserved
    # Classifications are preserved.
    assert "Hero" in reserved
    assert "Princess" in reserved
    # Keywords appear in both cases (card text is inconsistent).
    assert "Shift" in reserved
    assert "shift" in reserved


def test_reserved_tokens_are_not_split_by_bpe(tmp_path: Path) -> None:
    reserved = ["Tinker Bell", "{I}", "Hero"]
    tok = train_tokeniser(
        _SAMPLE_TEXTS + ["Tinker Bell gains Rush when you have a Hero character in play."],
        out_path=tmp_path / "tok.json",
        vocab_size=500,
        reserved_tokens=reserved,
    )
    # Each reserved token has its own id, and encoding the raw token
    # produces exactly that id (not a multi-piece breakdown).
    for value in reserved:
        id_ = tok.token_to_id(value)
        assert id_ is not None, value
        enc = tok.encode(value, add_special_tokens=False)
        assert enc.ids == [id_], (value, enc.ids)


def test_reserved_tokens_kept_atomic_inside_real_card_text(tmp_path: Path) -> None:
    tok = train_tokeniser(
        _SAMPLE_TEXTS,
        out_path=tmp_path / "tok.json",
        vocab_size=500,
        reserved_tokens=("Elsa", "{I}"),
    )
    elsa_id = tok.token_to_id("Elsa")
    ink_id = tok.token_to_id("{I}")
    enc = tok.encode("Shift 4 (You may pay 4 {I} to play this on top of one of your characters named Elsa.)")
    assert elsa_id in enc.ids
    assert ink_id in enc.ids
    # And each appears only once — not doubly-emitted from a fallback path.
    assert enc.ids.count(elsa_id) == 1


def test_duplicate_reserved_tokens_are_harmless(tmp_path: Path) -> None:
    tok = train_tokeniser(
        _SAMPLE_TEXTS,
        out_path=tmp_path / "tok.json",
        vocab_size=500,
        reserved_tokens=["Elsa", "Elsa", PAD_TOKEN],  # duplicates + a core special
    )
    # The core special is unaffected; the duplicate is deduped.
    assert tok.token_to_id(PAD_TOKEN) == 0
    assert tok.token_to_id("Elsa") is not None
