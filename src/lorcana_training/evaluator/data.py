"""Evaluator training data + curriculum negative sampling.

Every training example is a (partial_deck, candidate_card, label) triple.
Positives come from the same masked-set-completion setup the proposal
net uses: remove one copy of a real deck card, ask the evaluator
whether that card is plausible as the next addition.

Negatives are where the work is. A random-any-card negative makes the
task "is this card even close to the right inks?" — trivial. The
curriculum ramps through three phases, each harder than the last:

1. **warmup** (``CurriculumPhase.RANDOM_IN_INK``):
   draw uniformly from the pool of cards whose ink set overlaps the
   deck's inks. Teaches "right inks, plausible cost band."

2. **curve-matching** (``CurriculumPhase.CURVE_MATCHED``):
   draw from cards matching both the removed card's cost bucket and
   the deck's inks. Removes the easy "cost is obviously wrong" cue
   and forces the model to learn card-identity signal.

3. **local-swap** (``CurriculumPhase.LOCAL_SWAP``):
   swap in a card that occupies the analogous slot in a *different*
   real deck with the same ink pair (same cost bucket, different
   deck). Forces deck-internal synergy learning — "this particular
   partial wants *this* card, not the meta-baseline card for the
   same inks + cost."

Sampling from a real card pool rather than freshly generating a
candidate id means every negative is a legal, in-vocab card. The
evaluator's job isn't to reject out-of-vocab ids — that's what the
legality mask does at inference — so training the model on those
wastes capacity.

The :class:`NegativeSampler` precomputes all the indexes each phase
needs (per-ink card lists, per-(cost, ink) card lists, per-ink-pair
deck lists) so ``__getitem__`` is a fast lookup rather than a scan.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from ..proposal.data import Deck, ink_multihot, sample_partial


class CurriculumPhase(str, Enum):
    """Negative-sampling phase. See module docstring for the full
    motivation; the values are the same strings the CLI accepts."""

    RANDOM_IN_INK = "random_in_ink"
    CURVE_MATCHED = "curve_matched"
    LOCAL_SWAP = "local_swap"


# Cardinal Lorcana ink list in the same order as ``cards.features._INKS``
# (lowercased). Index matches the one-hot slot in the feature tensor.
_INK_ORDER: tuple[str, ...] = (
    "amber",
    "amethyst",
    "emerald",
    "ruby",
    "sapphire",
    "steel",
)


@dataclass(frozen=True, slots=True)
class CardIndex:
    """Compact per-card lookup needed by the curriculum samplers.

    All arrays are indexed by logical card id 0..vocab_size. Row 0 is
    the PAD placeholder and never appears as a negative; the samplers
    filter it out.

    Attributes:
      ``cost``           card's primary cost bucket (argmax of the
                         cost one-hot). int, -1 for PAD.
      ``ink_mask``       6-bit mask of inks the card belongs to. int.
      ``by_ink``         for each of the 6 inks, the tensor of card
                         ids that contain that ink. Sorted asc.
      ``by_ink_cost``    ``by_ink_cost[ink][cost]`` → card ids with
                         exactly that (ink, cost) combo. Sparse dict
                         because not every bucket is populated.
      ``all_cards``      tensor of every non-PAD card id. Used as the
                         fallback pool when a (ink, cost) bucket is
                         empty.
    """

    cost: torch.Tensor  # (V+1,) int64
    ink_mask: torch.Tensor  # (V+1,) int64
    by_ink: tuple[torch.Tensor, ...]  # len 6, each 1-D
    by_ink_cost: dict[tuple[int, int], torch.Tensor]
    all_cards: torch.Tensor  # (V,) int64, all non-PAD ids

    @property
    def vocab_size(self) -> int:
        return int(self.cost.shape[0]) - 1


def build_card_index(
    *,
    card_features_path: Path,
    feature_schema_path: Path,
) -> CardIndex:
    """Construct the :class:`CardIndex` from prepare's outputs."""
    schema: dict[str, Any] = json.loads(feature_schema_path.read_text(encoding="utf8"))
    cost_start, cost_len = schema["slices"]["cost"]
    inks_start, inks_len = schema["slices"]["inks"]
    if inks_len != len(_INK_ORDER):
        raise ValueError(
            f"feature_schema inks dim {inks_len} != expected {len(_INK_ORDER)}",
        )

    features = load_file(str(card_features_path))["card_features"]
    if features.dim() != 2:
        raise ValueError(f"card_features must be 2-D; got {tuple(features.shape)}")
    vocab_plus_pad = features.shape[0]

    cost_slice = features[:, cost_start : cost_start + cost_len]
    # argmax handles PAD rows (all zeros) too; we overwrite row 0's cost
    # below because it's semantically undefined for PAD.
    cost = cost_slice.argmax(dim=1).long()
    cost[0] = -1

    ink_slice = features[:, inks_start : inks_start + inks_len]  # float 0/1
    # Pack each row's 6 ink bits into an int64 mask for fast bit-ops.
    weights = torch.tensor([1 << i for i in range(inks_len)], dtype=torch.long)
    ink_mask = (ink_slice.long() * weights).sum(dim=1)

    # Per-ink card id lists. We sort for determinism + easier debugging.
    all_ids = torch.arange(1, vocab_plus_pad, dtype=torch.long)
    by_ink: list[torch.Tensor] = []
    for ink_idx in range(inks_len):
        bit = 1 << ink_idx
        members = all_ids[(ink_mask[1:] & bit) != 0]
        by_ink.append(members)

    # (ink, cost) → card ids. Skip the PAD row (index 0).
    by_ink_cost: dict[tuple[int, int], torch.Tensor] = {}
    for ink_idx in range(inks_len):
        bit = 1 << ink_idx
        for cost_val in range(cost_len):
            # `cost` contains the argmax bucket, which is what we key on
            # because that's the only card-cost representation downstream
            # models (and the CardEncoder's struct features) share.
            mask = ((ink_mask & bit) != 0) & (cost == cost_val)
            mask[0] = False  # never include PAD
            members = torch.nonzero(mask, as_tuple=False).squeeze(1)
            if members.numel() > 0:
                by_ink_cost[(ink_idx, cost_val)] = members

    return CardIndex(
        cost=cost,
        ink_mask=ink_mask,
        by_ink=tuple(by_ink),
        by_ink_cost=by_ink_cost,
        all_cards=all_ids,
    )


def _inks_to_bits(inks: tuple[str, ...]) -> int:
    """Convert a deck's canonical ink list to the matching 6-bit mask."""
    bits = 0
    for name in inks:
        try:
            bits |= 1 << _INK_ORDER.index(name.lower())
        except ValueError:
            # Unknown ink — silently ignore (same as proposal loader).
            continue
    return bits


class NegativeSampler:
    """Draws a negative candidate id given a (deck, removed_card, phase).

    Thread-unsafe by design: each ``__call__`` uses the caller-provided
    ``rng`` (a per-sample ``random.Random`` seeded deterministically
    from the dataset seed + index) so multi-worker DataLoaders don't
    share state. Falls back to a broader pool when a tight bucket is
    empty, and never returns PAD.
    """

    def __init__(
        self,
        *,
        card_index: CardIndex,
        ink_pair_to_decks: dict[tuple[str, ...], list[int]] | None = None,
        decks: list[Deck] | None = None,
    ) -> None:
        self._idx = card_index
        self._ink_pair_to_decks = ink_pair_to_decks or {}
        self._decks = decks or []

    def sample(
        self,
        *,
        phase: CurriculumPhase,
        deck_inks: tuple[str, ...],
        removed_card: int,
        exclude: set[int],
        rng: random.Random,
    ) -> int:
        """Return a single negative card id matching ``phase``'s policy."""
        if phase is CurriculumPhase.RANDOM_IN_INK:
            return self._random_in_ink(deck_inks, exclude, rng)
        if phase is CurriculumPhase.CURVE_MATCHED:
            return self._curve_matched(deck_inks, removed_card, exclude, rng)
        if phase is CurriculumPhase.LOCAL_SWAP:
            return self._local_swap(deck_inks, removed_card, exclude, rng)
        raise ValueError(f"unknown curriculum phase: {phase}")

    # -- Phase 1 ----------------------------------------------------

    def _random_in_ink(
        self,
        deck_inks: tuple[str, ...],
        exclude: set[int],
        rng: random.Random,
    ) -> int:
        pool = self._pool_for_inks(deck_inks)
        return self._sample_from_pool(pool, exclude, rng, fallback=self._idx.all_cards)

    # -- Phase 2 ----------------------------------------------------

    def _curve_matched(
        self,
        deck_inks: tuple[str, ...],
        removed_card: int,
        exclude: set[int],
        rng: random.Random,
    ) -> int:
        cost = int(self._idx.cost[removed_card].item())
        # Union of (ink, cost) buckets for every ink the deck has; if
        # nothing matches we back off to "random in ink" which is
        # already harder than "random any card" and avoids a dead end.
        buckets: list[torch.Tensor] = []
        for ink_idx in self._deck_ink_indexes(deck_inks):
            bucket = self._idx.by_ink_cost.get((ink_idx, cost))
            if bucket is not None:
                buckets.append(bucket)
        if not buckets:
            return self._random_in_ink(deck_inks, exclude, rng)
        pool: torch.Tensor = (
            buckets[0]
            if len(buckets) == 1
            else torch.cat(buckets).unique()  # type: ignore[no-untyped-call]
        )
        return self._sample_from_pool(pool, exclude, rng, fallback=self._pool_for_inks(deck_inks))

    # -- Phase 3 ----------------------------------------------------

    def _local_swap(
        self,
        deck_inks: tuple[str, ...],
        removed_card: int,
        exclude: set[int],
        rng: random.Random,
    ) -> int:
        """Pick a card at the analogous cost slot from a different
        deck with matching inks. Falls back to curve-matched (which
        itself falls back to random-in-ink) if no suitable deck
        exists.
        """
        # Look up peer decks by canonical ink tuple.
        key = tuple(sorted(ink.lower() for ink in deck_inks))
        peer_decks = self._ink_pair_to_decks.get(key, [])
        if len(peer_decks) < 2:
            return self._curve_matched(deck_inks, removed_card, exclude, rng)

        cost = int(self._idx.cost[removed_card].item())
        # Shuffle peers once, pick the first one that yields a card
        # with the target cost. Capping attempts keeps the worst case
        # bounded even if no peer has a same-cost card.
        order = list(peer_decks)
        rng.shuffle(order)
        for deck_idx in order[:12]:
            deck = self._decks[deck_idx]
            candidates = [idx for idx, _ in deck.cards if int(self._idx.cost[idx]) == cost]
            candidates = [c for c in candidates if c not in exclude]
            if candidates:
                return rng.choice(candidates)
        return self._curve_matched(deck_inks, removed_card, exclude, rng)

    # -- Shared helpers --------------------------------------------

    def _deck_ink_indexes(self, deck_inks: tuple[str, ...]) -> list[int]:
        out: list[int] = []
        for ink in deck_inks:
            try:
                out.append(_INK_ORDER.index(ink.lower()))
            except ValueError:
                continue
        return out

    def _pool_for_inks(self, deck_inks: tuple[str, ...]) -> torch.Tensor:
        """Union of per-ink pools for every ink the deck has."""
        pools = [self._idx.by_ink[i] for i in self._deck_ink_indexes(deck_inks)]
        if not pools:
            return self._idx.all_cards
        if len(pools) == 1:
            return pools[0]
        merged: torch.Tensor = torch.cat(pools).unique()  # type: ignore[no-untyped-call]
        return merged

    def _sample_from_pool(
        self,
        pool: torch.Tensor,
        exclude: set[int],
        rng: random.Random,
        *,
        fallback: torch.Tensor,
        max_attempts: int = 16,
    ) -> int:
        # We don't materialise the ``pool \ exclude`` set each draw;
        # rejection sampling is cheaper and lets us keep ``pool`` as a
        # flat int64 tensor (shared across calls without copying).
        size = int(pool.numel())
        if size == 0:
            pool = fallback
            size = int(pool.numel())
        if size == 0:  # both empty → last-resort sentinel
            return 1
        for _ in range(max_attempts):
            card_id = int(pool[rng.randrange(size)].item())
            if card_id != 0 and card_id not in exclude:
                return card_id
        # If we couldn't find anything after max_attempts, broaden to
        # the fallback pool. Keep applying ``exclude`` — the widened
        # pool still shouldn't return a card already in the deck;
        # otherwise the positive and negative share a target id, which
        # trains the evaluator toward "contradict yourself".
        fallback_size = int(fallback.numel())
        if fallback_size > 0:
            for _ in range(max_attempts):
                card_id = int(fallback[rng.randrange(fallback_size)].item())
                if card_id != 0 and card_id not in exclude:
                    return card_id
            # All pools exhausted — pick any non-PAD fallback id even
            # if it's excluded. The loss still learns something because
            # the deck is the conditioning context, not the label.
            card_id = int(fallback[rng.randrange(fallback_size)].item())
            return card_id if card_id != 0 else 1
        return 1


def build_ink_pair_to_decks(decks: list[Deck]) -> dict[tuple[str, ...], list[int]]:
    """Group deck indices by sorted ink-pair key for local-swap lookup."""
    out: dict[tuple[str, ...], list[int]] = {}
    for i, deck in enumerate(decks):
        key = tuple(sorted(ink.lower() for ink in deck.inks))
        out.setdefault(key, []).append(i)
    return out


@dataclass(frozen=True, slots=True)
class EvaluatorSample:
    partial_ids: torch.Tensor  # (N,) int64
    candidate_id: torch.Tensor  # scalar int64
    label: torch.Tensor  # scalar float {0.0, 1.0}


class EvaluatorDataset(Dataset[EvaluatorSample]):
    """Positive + negative pairs for the evaluator.

    Each deck contributes ``2 × samples_per_deck`` examples per epoch:
    one positive + one curriculum negative per mask draw. The caller
    sets the active :class:`CurriculumPhase` on the dataset (via
    :meth:`set_phase`) before each epoch. Keeping the phase on the
    dataset (rather than on the sample) lets the DataLoader shuffle
    across decks while every draw in the epoch uses the same phase.

    ``ink_multihot`` is ignored here — the model has no ink head.
    The :func:`lorcana_training.proposal.data.ink_multihot` import is
    deliberate so anyone repurposing this dataset for an ink-aware
    downstream model doesn't have to duplicate the encoder.
    """

    def __init__(
        self,
        decks: list[Deck],
        *,
        card_index: CardIndex,
        samples_per_deck: int = 12,
        initial_phase: CurriculumPhase = CurriculumPhase.RANDOM_IN_INK,
        seed: int | None = None,
    ) -> None:
        if not decks:
            raise ValueError("EvaluatorDataset requires at least one deck.")
        if samples_per_deck <= 0:
            raise ValueError("samples_per_deck must be positive.")
        self._decks = list(decks)
        self._card_index = card_index
        self._samples_per_deck = samples_per_deck
        self._seed = seed
        self._phase = initial_phase
        ink_pair_to_decks = build_ink_pair_to_decks(self._decks)
        self._sampler = NegativeSampler(
            card_index=card_index,
            ink_pair_to_decks=ink_pair_to_decks,
            decks=self._decks,
        )
        # Ignore the imported helper at module level — we reference it
        # here so linters don't flag the import as unused. Downstream
        # code can call ``ink_multihot`` on a sample's deck without
        # re-importing from ``proposal.data``.
        _ = ink_multihot

    def __len__(self) -> int:
        # 2 examples per mask (one positive + one negative).
        return len(self._decks) * self._samples_per_deck * 2

    def set_phase(self, phase: CurriculumPhase) -> None:
        self._phase = phase

    @property
    def phase(self) -> CurriculumPhase:
        return self._phase

    def __getitem__(self, index: int) -> EvaluatorSample:
        # Index layout: even indexes = positive, odd indexes = negative.
        # Pairing them (rather than interleaving elsewhere) keeps the
        # (positive, negative) draws that share a partial contiguous,
        # which is useful for debugging.
        is_negative = index % 2 == 1
        pair_index = index // 2
        deck_index = pair_index // self._samples_per_deck
        rng = random.Random(None if self._seed is None else self._seed ^ index ^ 0x51ED270F)

        deck = self._decks[deck_index]
        remaining, removed = sample_partial(deck, rng)
        partial_ids = (
            torch.tensor(remaining, dtype=torch.long)
            if remaining
            else torch.zeros(1, dtype=torch.long)
        )

        if not is_negative:
            return EvaluatorSample(
                partial_ids=partial_ids,
                candidate_id=torch.tensor(removed, dtype=torch.long),
                label=torch.tensor(1.0, dtype=torch.float32),
            )

        # Exclude every card already in the deck from the negative
        # pool: otherwise a "different card" draw could land on a card
        # the deck also plays, which is actually a valid next pick.
        deck_card_ids = {idx for idx, _ in deck.cards}
        negative = self._sampler.sample(
            phase=self._phase,
            deck_inks=deck.inks,
            removed_card=removed,
            exclude=deck_card_ids,
            rng=rng,
        )
        return EvaluatorSample(
            partial_ids=partial_ids,
            candidate_id=torch.tensor(negative, dtype=torch.long),
            label=torch.tensor(0.0, dtype=torch.float32),
        )


def collate_evaluator(batch: list[EvaluatorSample]) -> dict[str, torch.Tensor]:
    """Right-pad partials with PAD=0 and stack candidates + labels."""
    if not batch:
        raise ValueError("collate_evaluator received an empty batch.")
    max_len = max(sample.partial_ids.shape[0] for sample in batch)
    partial_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, sample in enumerate(batch):
        n = sample.partial_ids.shape[0]
        partial_ids[i, :n] = sample.partial_ids
    candidate_ids = torch.stack([s.candidate_id for s in batch], dim=0)
    labels = torch.stack([s.label for s in batch], dim=0)
    return {
        "partial_ids": partial_ids,
        "candidate_ids": candidate_ids,
        "labels": labels,
    }
