"""Tests for pretrain/data.py helpers.

The full pretrain run is covered by an opt-in live smoke under
``RUN_NETWORK_TESTS=1`` in CI; these are fast unit tests for the
split + tokeniser + dataset wiring against tiny fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.numpy import save_file

from lorcana_training.cards.features import build_feature_schema, build_features
from lorcana_training.cards.logical import build_logical_cards
from lorcana_training.cards.vocab import build_vocab
from lorcana_training.pretrain.data import (
    CardPretrainDataset,
    PreparedPaths,
    build_pretrain_dataset,
    build_pretrain_tokeniser,
    collate,
    split_indices,
)
from lorcana_training.schemas.generated.card_set import CardSet


def _card(
    id_: str,
    name: str,
    *,
    text: str = "",
    classifications: tuple[str, ...] = ("Storyborn",),
    keywords: tuple[str, ...] = (),
) -> dict:
    return {
        "id": id_,
        "name": name,
        "version": "Test",
        "setCode": "1",
        "cardNumber": int(id_.split("_")[-1]),
        "cost": 3,
        "inkwell": True,
        "inks": ["Amber"],
        "types": ["Character"],
        "classifications": list(classifications),
        "keywords": list(keywords),
        "text": text,
        "flavor": None,
        "imageUrl": f"https://example.test/{id_}.avif",
        "legality": "legal",
        "lore": 2,
        "strength": 3,
        "willpower": 3,
        "moveCost": None,
    }


def _fixture_card_set() -> CardSet:
    cards = [
        _card(
            f"crd_{i:02d}",
            f"Character{i}",
            text=f"Rush {i}. When you play this character, gain {i} lore.",
            keywords=("Rush",),
        )
        for i in range(20)
    ]
    return CardSet.model_validate(
        {"cardSetVersion": "sha256:t", "fetchedAt": "2026-05-13T00:00:00Z", "cards": cards}
    )


def test_split_indices_is_deterministic_and_covers_all() -> None:
    cs = _fixture_card_set()
    logical = build_logical_cards(cs)
    train, heldout = split_indices(logical, heldout_ratio=0.2, seed="test")
    assert sorted(train + heldout) == list(range(len(logical.cards)))
    # Deterministic: same seed -> same split.
    train_b, heldout_b = split_indices(logical, heldout_ratio=0.2, seed="test")
    assert train == train_b and heldout == heldout_b
    # Different seed -> almost-certainly different partition.
    train_c, _ = split_indices(logical, heldout_ratio=0.2, seed="other")
    assert train != train_c


def test_dataset_produces_fixed_shape_rows(tmp_path: Path) -> None:
    cs = _fixture_card_set()
    logical = build_logical_cards(cs)
    vocab = build_vocab(logical)
    schema = build_feature_schema(logical.cards)
    features = build_features(vocab, schema)
    feat_path = tmp_path / "card_features.safetensors"
    save_file({"card_features": features}, str(feat_path))

    tokeniser = build_pretrain_tokeniser(logical, out_path=tmp_path / "tok.json", vocab_size=500)
    data = build_pretrain_dataset(
        PreparedPaths(
            vocab=tmp_path / "vocab.json",
            card_features=feat_path,
            feature_schema=tmp_path / "feature_schema.json",
        ),
        logical_cards=logical,
        vocab=vocab,
        schema=schema,
        tokeniser=tokeniser,
        heldout_ratio=0.2,
    )

    ds = CardPretrainDataset(data, indices=data.train_indices, max_positions=64)
    row = ds[0]
    assert row["token_ids"].shape == (64,)
    assert row["struct_features"].shape == (schema.dim,)
    assert row["token_ids"].dtype == torch.long

    batch = collate([ds[i] for i in range(3)])
    assert batch["token_ids"].shape == (3, 64)
    assert batch["struct_features"].shape == (3, schema.dim)


def test_features_row_count_mismatch_raises(tmp_path: Path) -> None:
    cs = _fixture_card_set()
    logical = build_logical_cards(cs)
    vocab = build_vocab(logical)
    schema = build_feature_schema(logical.cards)
    # Bad features tensor (missing rows).
    wrong = np.zeros((5, schema.dim), dtype=np.float32)
    feat_path = tmp_path / "bad.safetensors"
    save_file({"card_features": wrong}, str(feat_path))
    tokeniser = build_pretrain_tokeniser(logical, out_path=tmp_path / "tok.json", vocab_size=500)
    with pytest.raises(ValueError, match="features tensor row count"):
        build_pretrain_dataset(
            PreparedPaths(
                vocab=tmp_path / "vocab.json",
                card_features=feat_path,
                feature_schema=tmp_path / "feature_schema.json",
            ),
            logical_cards=logical,
            vocab=vocab,
            schema=schema,
            tokeniser=tokeniser,
        )
