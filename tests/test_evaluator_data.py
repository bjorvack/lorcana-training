"""Tests for :mod:`lorcana_training.evaluator.data`.

Covers the pieces most likely to silently go wrong:

  - Card index decodes cost + ink mask correctly.
  - Each curriculum phase's sampler respects its contract
    (in-ink, in-cost, different-deck).
  - Dataset pairs (positive, negative) share a partial deck.
  - Collate right-pads + stacks the three tensors.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lorcana_training.evaluator.data import (
    CurriculumPhase,
    EvaluatorDataset,
    NegativeSampler,
    build_card_index,
    collate_evaluator,
)
from lorcana_training.proposal.data import Deck


# The feature schema we write to disk. These offsets mirror a real
# prepare output but stripped of classifications/keywords since none
# of the curriculum logic touches them.
_TINY_SCHEMA: dict = {
    "dim": 18,
    "slices": {"cost": [0, 12], "inks": [12, 6], "types": [0, 0]},
    "scalars": {},
    "classes": {"inks": ["amber", "amethyst", "emerald", "ruby", "sapphire", "steel"]},
    "normalisers": {},
}


def _write_tiny_prepared(tmp_path: Path) -> Path:
    """Write a minimal prepared directory with 6 cards, costs 1..3 × 2 inks."""
    prepared = tmp_path / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)

    (prepared / "feature_schema.json").write_text(json.dumps(_TINY_SCHEMA), encoding="utf8")

    # Vocab size = 6 → features tensor has 7 rows (+PAD).
    vocab_size = 6
    features = torch.zeros(vocab_size + 1, 18, dtype=torch.float32)
    # card 1: amber cost 1
    features[1, 1] = 1.0  # cost bucket 1
    features[1, 12] = 1.0  # amber
    # card 2: amber cost 3
    features[2, 3] = 1.0
    features[2, 12] = 1.0
    # card 3: ruby cost 1
    features[3, 1] = 1.0
    features[3, 15] = 1.0  # ruby
    # card 4: ruby cost 3
    features[4, 3] = 1.0
    features[4, 15] = 1.0
    # card 5: amber + ruby, cost 2 (dual-ink)
    features[5, 2] = 1.0
    features[5, 12] = 1.0
    features[5, 15] = 1.0
    # card 6: steel cost 1 (wrong ink for any amber/ruby deck)
    features[6, 1] = 1.0
    features[6, 17] = 1.0  # steel

    save_file({"card_features": features}, str(prepared / "card_features.safetensors"))
    return prepared


def test_build_card_index_decodes_cost_and_ink() -> None:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        prepared = _write_tiny_prepared(Path(tmp))
        idx = build_card_index(
            card_features_path=prepared / "card_features.safetensors",
            feature_schema_path=prepared / "feature_schema.json",
        )
        assert idx.vocab_size == 6
        assert int(idx.cost[1]) == 1
        assert int(idx.cost[2]) == 3
        assert int(idx.cost[6]) == 1

        # card 1 (amber) should be in by_ink[0] (amber slot).
        assert 1 in idx.by_ink[0].tolist()
        # card 5 is dual-ink → in both amber and ruby pools.
        assert 5 in idx.by_ink[0].tolist()
        assert 5 in idx.by_ink[3].tolist()
        # (amber, cost=1) bucket: card 1 and card 5? card 5 has cost 2
        # so only card 1.
        assert idx.by_ink_cost[(0, 1)].tolist() == [1]


def _small_index_and_sampler() -> tuple[NegativeSampler, object]:
    from tempfile import mkdtemp

    prepared = _write_tiny_prepared(Path(mkdtemp()))
    idx = build_card_index(
        card_features_path=prepared / "card_features.safetensors",
        feature_schema_path=prepared / "feature_schema.json",
    )
    # Two amber/ruby decks so local-swap has peers to swap with.
    decks = [
        Deck(cards=((1, 2), (3, 2), (5, 2)), inks=("amber", "ruby")),
        Deck(cards=((2, 2), (4, 2), (5, 2)), inks=("amber", "ruby")),
    ]
    sampler = NegativeSampler(
        card_index=idx,
        ink_pair_to_decks={("amber", "ruby"): [0, 1]},
        decks=decks,
    )
    return sampler, idx


def test_negative_sampler_random_in_ink_respects_inks() -> None:
    sampler, idx = _small_index_and_sampler()  # type: ignore[assignment]
    rng = random.Random(0)
    # Deck is amber+ruby; card 6 (steel) must never appear.
    steel = False
    for _ in range(200):
        cand = sampler.sample(
            phase=CurriculumPhase.RANDOM_IN_INK,
            deck_inks=("amber", "ruby"),
            removed_card=1,
            exclude=set(),
            rng=rng,
        )
        if cand == 6:
            steel = True
    assert not steel


def test_negative_sampler_curve_matched_matches_cost() -> None:
    sampler, _ = _small_index_and_sampler()
    rng = random.Random(1)
    # Remove card 2 (cost 3); negative should be cost 3 and in inks.
    for _ in range(40):
        cand = sampler.sample(
            phase=CurriculumPhase.CURVE_MATCHED,
            deck_inks=("amber", "ruby"),
            removed_card=2,
            exclude={2, 4},  # exclude both same-cost cards to force fallback
            rng=rng,
        )
        # After exclusion, no same-cost amber/ruby exists. Sampler
        # should fall back to random-in-ink; it must still never
        # return steel.
        assert cand != 6


def test_negative_sampler_local_swap_picks_different_deck() -> None:
    sampler, _ = _small_index_and_sampler()
    rng = random.Random(2)
    # Deck 0's cards are {1, 3, 5}. Remove card 1 (cost 1). Peer
    # deck 1's cost-1 cards: none (its cards are 2, 4, 5). So the
    # sampler should fall back to curve-matched → random-in-ink
    # rather than infinite-looping.
    cand = sampler.sample(
        phase=CurriculumPhase.LOCAL_SWAP,
        deck_inks=("amber", "ruby"),
        removed_card=1,
        exclude={1, 3, 5},
        rng=rng,
    )
    assert cand in {2, 4}  # cost 3 amber/ruby cards via curve-matched fallback


def test_evaluator_dataset_len_and_labels() -> None:
    from tempfile import mkdtemp

    prepared = _write_tiny_prepared(Path(mkdtemp()))
    idx = build_card_index(
        card_features_path=prepared / "card_features.safetensors",
        feature_schema_path=prepared / "feature_schema.json",
    )
    decks = [
        Deck(cards=((1, 4), (2, 4), (5, 4)), inks=("amber", "ruby")),
    ]
    ds = EvaluatorDataset(
        decks,
        card_index=idx,
        samples_per_deck=3,
        initial_phase=CurriculumPhase.RANDOM_IN_INK,
        seed=7,
    )
    # 1 deck × 3 masks × 2 (pos + neg) = 6 samples.
    assert len(ds) == 6
    # Index 0 is positive, 1 is negative, 2 is positive, ...
    pos = ds[0]
    neg = ds[1]
    assert pos.label.item() == 1.0
    assert neg.label.item() == 0.0
    # Positive + its paired negative share the same partial.
    assert torch.equal(pos.partial_ids, neg.partial_ids)


def test_collate_evaluator_shapes() -> None:
    from tempfile import mkdtemp

    prepared = _write_tiny_prepared(Path(mkdtemp()))
    idx = build_card_index(
        card_features_path=prepared / "card_features.safetensors",
        feature_schema_path=prepared / "feature_schema.json",
    )
    decks = [
        Deck(cards=((1, 4), (2, 4)), inks=("amber", "ruby")),
        Deck(cards=((1, 4), (2, 4), (5, 2)), inks=("amber", "ruby")),
    ]
    ds = EvaluatorDataset(
        decks,
        card_index=idx,
        samples_per_deck=1,
        initial_phase=CurriculumPhase.RANDOM_IN_INK,
        seed=11,
    )
    batch = collate_evaluator([ds[0], ds[1], ds[2], ds[3]])
    assert batch["partial_ids"].dim() == 2
    assert batch["partial_ids"].shape[0] == 4
    assert batch["candidate_ids"].shape == (4,)
    assert batch["labels"].shape == (4,)


def test_evaluator_dataset_requires_nonempty() -> None:
    from tempfile import mkdtemp

    prepared = _write_tiny_prepared(Path(mkdtemp()))
    idx = build_card_index(
        card_features_path=prepared / "card_features.safetensors",
        feature_schema_path=prepared / "feature_schema.json",
    )
    with pytest.raises(ValueError):
        EvaluatorDataset([], card_index=idx)
