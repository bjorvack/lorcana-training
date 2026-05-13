"""Tests for ``lorcana_training.proposal.data`` (dataset + collate).

Covers:

  - JSONL loader drops malformed rows, preserves well-formed ones.
  - Ink multi-hot encodes the canonical order.
  - Target distribution normalises to 1 and respects copy counts.
  - ``sample_partial`` removes exactly one copy; multiplicity-weighted.
  - ProposalDataset produces stable shapes; len = decks * k.
  - Collate right-pads with 0 and stacks targets + ink vectors.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

from lorcana_training.proposal.data import (
    INK_ORDER,
    Deck,
    ProposalDataset,
    TargetMode,
    collate_proposal,
    ink_multihot,
    load_decks_jsonl,
    sample_partial,
    target_distribution_from_deck,
)


def _deck(cards: list[tuple[int, int]], inks: list[str]) -> Deck:
    return Deck(cards=tuple(sorted(cards)), inks=tuple(inks))


def test_ink_multihot_matches_canonical_order() -> None:
    v = ink_multihot(["amber", "ruby"])
    assert v.shape == (6,)
    assert v[INK_ORDER.index("amber")].item() == 1.0
    assert v[INK_ORDER.index("ruby")].item() == 1.0
    assert v.sum().item() == 2.0


def test_ink_multihot_case_insensitive() -> None:
    v_lower = ink_multihot(["amber", "ruby"])
    v_upper = ink_multihot(["AMBER", "Ruby"])
    assert torch.equal(v_lower, v_upper)


def test_ink_multihot_skips_unknown() -> None:
    v = ink_multihot(["amber", "teal"])
    assert v.sum().item() == 1.0


def test_target_distribution_normalises_and_zeros_unseen() -> None:
    deck = _deck([(5, 4), (7, 4), (11, 2)], ["amber", "ruby"])
    vocab_size = 20
    target = target_distribution_from_deck(deck, vocab_size)
    assert target.shape == (vocab_size + 1,)
    assert target.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert target[5].item() == pytest.approx(0.4)
    assert target[7].item() == pytest.approx(0.4)
    assert target[11].item() == pytest.approx(0.2)
    # The PAD slot and every other index must be zero.
    assert target[0].item() == 0.0
    assert target[3].item() == 0.0


def test_sample_partial_removes_exactly_one_copy() -> None:
    deck = _deck([(5, 4), (7, 4), (11, 2)], ["amber", "ruby"])
    rng = random.Random(0)
    partial, removed = sample_partial(deck, rng)
    assert len(partial) == deck.total_copies - 1
    assert 0 not in partial  # never PAD
    # The removed id must itself be non-PAD and present in the original deck.
    assert removed != 0
    assert removed in {idx for idx, _ in deck.cards}


def test_sample_partial_is_multiplicity_weighted() -> None:
    """4-copy cards should be removed ~4x more often than 1-copy cards."""
    deck = _deck([(5, 4), (11, 1)], ["amber", "ruby"])
    total = deck.total_copies  # 5
    rng = random.Random(1234)
    removed_from_5 = 0
    n_trials = 2000
    for _ in range(n_trials):
        _, removed = sample_partial(deck, rng)
        if removed == 5:
            removed_from_5 += 1
    # Expected fraction: 4/5 = 0.8. Allow a generous 3σ-ish tolerance
    # (binomial stddev for n=2000, p=0.8 is ~18, so ±0.03 on the
    # proportion is ~5σ).
    removal_rate_5 = removed_from_5 / n_trials
    assert abs(removal_rate_5 - (4.0 / total)) < 0.03


def test_load_decks_jsonl_drops_malformed(tmp_path: Path) -> None:
    p = tmp_path / "decks.jsonl"
    lines = [
        json.dumps({"cards": [[1, 4], [2, 4]], "inks": ["amber", "ruby"]}),
        json.dumps({"cards": [], "inks": ["amber"]}),  # empty cards
        json.dumps({"cards": [[3, 2]], "inks": []}),  # empty inks
        json.dumps({"cards": [[4, 4]], "inks": ["steel", "emerald"]}),
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf8")
    decks = load_decks_jsonl(p)
    assert len(decks) == 2
    assert decks[0].cards == ((1, 4), (2, 4))
    assert decks[1].cards == ((4, 4),)


def test_proposal_dataset_len_and_sample_shapes() -> None:
    decks = [
        _deck(
            [
                (5, 4),
                (7, 4),
                (11, 4),
                (12, 4),
                (13, 4),
                (14, 4),
                (15, 4),
                (16, 4),
                (17, 4),
                (18, 4),
                (19, 4),
                (20, 4),
                (21, 4),
                (22, 4),
                (23, 4),
            ],
            ["amber", "ruby"],
        ),
        _deck(
            [
                (5, 4),
                (7, 4),
                (11, 4),
                (12, 4),
                (13, 4),
                (14, 4),
                (15, 4),
                (16, 4),
                (17, 4),
                (18, 4),
                (19, 4),
                (20, 4),
                (21, 4),
                (22, 4),
                (23, 4),
            ],
            ["amber", "steel"],
        ),
    ]
    vocab_size = 30
    ds = ProposalDataset(decks, vocab_size=vocab_size, samples_per_deck=4, seed=0)
    # len = decks * k_pos
    assert len(ds) == 2 * 4
    sample = ds[3]
    assert sample.partial_ids.dim() == 1
    # 60 copies - 1 masked out.
    assert sample.partial_ids.shape[0] == 59
    assert sample.ink_multihot.shape == (6,)
    assert sample.target_distribution.shape == (vocab_size + 1,)
    assert sample.target_distribution.sum().item() == pytest.approx(1.0)


def test_proposal_dataset_is_deterministic_with_seed() -> None:
    decks = [
        _deck(
            [
                (1, 4),
                (2, 4),
                (3, 4),
                (4, 4),
                (5, 4),
                (6, 4),
                (7, 4),
                (8, 4),
                (9, 4),
                (10, 4),
                (11, 4),
                (12, 4),
                (13, 4),
                (14, 4),
                (15, 4),
            ],
            ["amber", "ruby"],
        ),
    ]
    ds_a = ProposalDataset(decks, vocab_size=20, samples_per_deck=3, seed=7)
    ds_b = ProposalDataset(decks, vocab_size=20, samples_per_deck=3, seed=7)
    for i in range(3):
        assert torch.equal(ds_a[i].partial_ids, ds_b[i].partial_ids)


def test_proposal_dataset_one_hot_target_mode_peaks_on_removed_card() -> None:
    """ONE_HOT_REMOVED puts all mass on exactly the removed card id."""
    decks = [
        _deck(
            [
                (1, 4),
                (2, 4),
                (3, 4),
                (4, 4),
                (5, 4),
                (6, 4),
                (7, 4),
                (8, 4),
                (9, 4),
                (10, 4),
                (11, 4),
                (12, 4),
                (13, 4),
                (14, 4),
                (15, 4),
            ],
            ["amber", "ruby"],
        ),
    ]
    ds = ProposalDataset(
        decks,
        vocab_size=20,
        samples_per_deck=5,
        target_mode=TargetMode.ONE_HOT_REMOVED,
        seed=42,
    )
    for i in range(5):
        sample = ds[i]
        target = sample.target_distribution
        # Exactly one non-zero, and its value is 1.
        assert target.sum().item() == pytest.approx(1.0)
        assert int((target > 0).sum().item()) == 1
        peaked_on = int(target.argmax().item())
        # The peaked-on card must be *one of* the deck's cards, and
        # specifically one that does NOT appear in the partial N times
        # that would match its original count (i.e. it's the card
        # that just lost a copy).
        original_count = {idx: c for idx, c in decks[0].cards}[peaked_on]
        partial_count = int((sample.partial_ids == peaked_on).sum().item())
        assert partial_count == original_count - 1


def test_collate_right_pads_with_zero() -> None:
    decks = [
        _deck([(5, 4), (7, 4), (11, 2)], ["amber", "ruby"]),
        _deck([(5, 4), (7, 4), (11, 4), (12, 4), (13, 4)], ["amber", "ruby"]),
    ]
    vocab_size = 30
    ds = ProposalDataset(decks, vocab_size=vocab_size, samples_per_deck=1, seed=0)
    batch = collate_proposal([ds[0], ds[1]])
    card_ids = batch["card_ids"]
    assert card_ids.dim() == 2
    assert card_ids.shape[0] == 2
    # max_len is 19 (20-1) for the larger deck; the smaller is
    # right-padded with 0s.
    # Smaller deck total copies = 4+4+2 = 10, partial 9.
    # Larger deck total = 20, partial 19.
    assert card_ids.shape[1] == 19
    # Row 0 tail is PAD.
    assert (card_ids[0, 9:] == 0).all()
    # Row 1 has no PAD.
    assert (card_ids[1, :] != 0).all()
    assert batch["ink_multihot"].shape == (2, 6)
    assert batch["target_distribution"].shape == (2, vocab_size + 1)
