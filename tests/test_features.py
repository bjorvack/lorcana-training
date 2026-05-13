"""Tests for the structured card features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file

from lorcana_training.cards.features import (
    build_feature_schema,
    build_features,
    write_features,
)
from lorcana_training.cards.logical import build_logical_cards
from lorcana_training.cards.vocab import build_vocab
from lorcana_training.schemas.generated.card_set import CardSet


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
    classifications: tuple[str, ...] = ("Storyborn",),
    keywords: tuple[str, ...] = (),
    text: str = "",
    lore: int | None = 2,
    strength: int | None = 3,
    willpower: int | None = 3,
    move_cost: int | None = None,
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
        "classifications": list(classifications),
        "keywords": list(keywords),
        "text": text,
        "flavor": None,
        "imageUrl": f"https://example.test/{id_}.avif",
        "legality": "legal",
        "lore": lore,
        "strength": strength,
        "willpower": willpower,
        "moveCost": move_cost,
    }


def _cs(*cards: dict) -> CardSet:
    return CardSet.model_validate(
        {
            "cardSetVersion": "sha256:test",
            "fetchedAt": "2026-05-13T00:00:00Z",
            "cards": list(cards),
        }
    )


def test_schema_slices_are_disjoint_and_cover_dim() -> None:
    cs = _cs(
        _card("crd_a", "A", keywords=("Shift", "Rush"), classifications=("Hero", "Storyborn")),
        _card("crd_b", "B", keywords=("Shift",), classifications=("Villain",)),
    )
    schema = build_feature_schema(build_logical_cards(cs).cards)
    positions: list[tuple[int, int]] = [
        schema.cost_slice,
        schema.inks_slice,
        schema.types_slice,
        schema.classifications_slice,
        schema.keywords_slice,
        (schema.lore_index, 1),
        (schema.strength_index, 1),
        (schema.willpower_index, 1),
        (schema.move_cost_index, 1),
        (schema.inkwell_index, 1),
    ]
    # Slices tile [0, dim) with no overlap.
    covered: list[int] = []
    for start, length in positions:
        covered.extend(range(start, start + length))
    assert sorted(covered) == list(range(schema.dim))
    # Classes discovered from the pool.
    assert set(schema.classifications) == {"Hero", "Storyborn", "Villain"}
    assert set(schema.keywords) == {"Rush", "Shift"}


def test_row_encoding_sets_expected_bits() -> None:
    cs = _cs(
        _card(
            "crd_a",
            "Mickey",
            version="True Friend",
            cost=3,
            inks=("Amber", "Ruby"),
            types=("Character",),
            classifications=("Storyborn", "Hero"),
            keywords=("Shift", "Rush"),
            lore=2,
            strength=3,
            willpower=5,
            inkwell=True,
        ),
    )
    logical = build_logical_cards(cs)
    vocab = build_vocab(logical)
    schema = build_feature_schema(logical.cards)
    features = build_features(vocab, schema)
    assert features.shape == (2, schema.dim)  # PAD + 1 card
    assert not features[0].any(), "pad row should be all zero"
    row = features[1]
    # Cost 3 → that cost bin is set, nothing else in the cost slice.
    cost_block = row[schema.cost_slice[0] : schema.cost_slice[0] + schema.cost_slice[1]]
    assert cost_block.argmax() == 3 and cost_block.sum() == 1.0
    # Two inks.
    ink_block = row[schema.inks_slice[0] : schema.inks_slice[0] + schema.inks_slice[1]]
    assert ink_block.sum() == 2.0
    # Character only.
    type_block = row[schema.types_slice[0] : schema.types_slice[0] + schema.types_slice[1]]
    assert type_block.sum() == 1.0
    # Two classifications, two keywords.
    cls_block = row[
        schema.classifications_slice[0] : schema.classifications_slice[0]
        + schema.classifications_slice[1]
    ]
    assert cls_block.sum() == 2.0
    kw_block = row[schema.keywords_slice[0] : schema.keywords_slice[0] + schema.keywords_slice[1]]
    assert kw_block.sum() == 2.0
    # Scalars normalised in [0, 1].
    assert 0.0 <= row[schema.lore_index] <= 1.0
    assert row[schema.inkwell_index] == 1.0


def test_location_move_cost_is_normalised() -> None:
    cs = _cs(
        _card(
            "crd_loc",
            "Atlantica",
            version="Concert Hall",
            types=("Location",),
            classifications=(),
            lore=None,
            strength=None,
            willpower=5,
            move_cost=3,
            inkwell=False,
        )
    )
    logical = build_logical_cards(cs)
    vocab = build_vocab(logical)
    schema = build_feature_schema(logical.cards)
    features = build_features(vocab, schema)
    row = features[1]
    # move_cost = 3, move_cost_max = 3 → 1.0
    assert row[schema.move_cost_index] == pytest.approx(1.0)
    # Location not inkable here.
    assert row[schema.inkwell_index] == 0.0


def test_write_features_round_trips(tmp_path: Path) -> None:
    cs = _cs(_card("crd_a", "A"), _card("crd_b", "B"))
    logical = build_logical_cards(cs)
    vocab = build_vocab(logical)
    schema = build_feature_schema(logical.cards)
    features = build_features(vocab, schema)
    paths = write_features(features, schema, out_dir=tmp_path)
    loaded = load_file(str(paths["card_features"]))["card_features"]
    assert np.array_equal(loaded, features)
    sdoc = json.loads(paths["feature_schema"].read_text())
    assert sdoc["dim"] == schema.dim
    # Class list is derived from pool content.
    assert sdoc["classes"]["classifications"] == list(schema.classifications)
