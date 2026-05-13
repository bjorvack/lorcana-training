"""Proposal-net training data.

Source: ``prepared/train.proposal.jsonl`` (and ``prepared/heldout.jsonl``),
each row representing one validated tournament deck:

    {"cards": [[logical_index, count], ...],
     "inks":  ["amber", "ruby"], ... }

The dataset turns each deck into a stream of *masked* examples:

    - Pick one card copy uniformly at random (i.e. copy-weighted).
    - Input: the 59 remaining card copies, as a multiset of logical ids.
    - Ink conditioning: a 6-dim multi-hot of the deck's ink pair.
    - Target: a distribution over vocab proportional to the original
      deck's card counts (``count / 60``). Every card that was in the
      deck gets non-zero mass; cards that weren't get zero. This
      matches DESIGN.md's "label smoothing for multisets" — a partial
      is consistent with any card from the original deck.

DESIGN.md specifies ``k_pos = 12`` masked examples per deck per epoch.
We expose that via ``samples_per_deck``: each ``__getitem__`` picks a
fresh random mask, and a ``RandomSampler`` is used so ``k_pos``
effectively falls out of the DataLoader's ``__len__ = len(decks) *
samples_per_deck`` plus shuffling. This way re-visiting the same deck
twice in a single epoch produces different masks.

The collate function right-pads variable-length partial decks to the
batch max — 59 for a full 60-card deck, less for decks at the legality
minimum. PAD = 0 matches the Transformer's key-padding mask convention
in :class:`lorcana_training.models.proposal.ProposalNet`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from ..models.proposal import INK_VECTOR_DIM


class TargetMode(str, Enum):
    """How to convert a deck + mask into a target distribution.

    - :attr:`FULL_DECK` (DESIGN.md "standard label-smoothing for
      multisets"): target[c] = count_in_full_deck(c) / 60, i.e. every
      card that was in the deck is a valid answer weighted by copies.
      Same target across all masks of a deck → the model effectively
      sees each deck as one (noisy-input, fixed-output) example.
    - :attr:`ONE_HOT_REMOVED`: target is a delta on the card that was
      removed in this specific mask. 12 masks per deck → 12 distinct
      (input, output) examples. Aggregated across masks the *marginal*
      target matches FULL_DECK, so the loss surface is the same; the
      per-sample gradient noise just acts as label-level
      regularisation that tends to fight overfitting on small sets.
    """

    FULL_DECK = "full_deck"
    ONE_HOT_REMOVED = "one_hot_removed"


# Canonical ink ordering. Must match
# ``cards.features._INKS`` (lowercased) so the multi-hot slot
# assignment lines up with the card encoder's struct features.
INK_ORDER: tuple[str, ...] = ("amber", "amethyst", "emerald", "ruby", "sapphire", "steel")
_INK_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(INK_ORDER)}


@dataclass(frozen=True, slots=True)
class Deck:
    """A validated deck after prepare's printing-to-logical rewrite.

    ``cards`` is a tuple of ``(logical_index, count)`` pairs sorted by
    index. Indices are guaranteed in ``1..vocab_size`` (0 is PAD).
    """

    cards: tuple[tuple[int, int], ...]
    inks: tuple[str, ...]

    @property
    def total_copies(self) -> int:
        return sum(c for _, c in self.cards)


def load_decks_jsonl(path: Path) -> list[Deck]:
    """Parse a prepare-output JSONL file into :class:`Deck` instances.

    Silently drops rows that don't have at least one card / ink — a
    prepared split should never contain those, but trusting inputs
    from disk blindly is how batch shapes end up NaN.
    """
    decks: list[Deck] = []
    with path.open("r", encoding="utf8") as f:
        for line in f:
            row = json.loads(line)
            raw_cards = row.get("cards") or []
            raw_inks = row.get("inks") or []
            cards = tuple(sorted((int(idx), int(count)) for idx, count in raw_cards))
            inks = tuple(ink.lower() for ink in raw_inks)
            if not cards or not inks:
                continue
            decks.append(Deck(cards=cards, inks=inks))
    return decks


def ink_multihot(inks: Iterable[str]) -> torch.Tensor:
    """Encode an ink pair/triple as a 6-dim {0, 1} multi-hot tensor."""
    out = torch.zeros(INK_VECTOR_DIM, dtype=torch.float32)
    for name in inks:
        idx = _INK_TO_INDEX.get(name.lower())
        if idx is None:
            # Unknown ink — skip rather than abort. A validated deck
            # should never contain one, but skipping avoids a training
            # crash if a future set adds an ink before the vocab is
            # bumped.
            continue
        out[idx] = 1.0
    return out


def target_distribution_from_deck(
    deck: Deck,
    vocab_size: int,
) -> torch.Tensor:
    """Build the dense target distribution for a deck.

    Matches DESIGN.md's "weight proportional to copies" rule with a
    simple multinomial interpretation: ``target[c] = count(c) /
    total``. Used identically across every mask of the same deck so
    the target doesn't depend on which card happened to be removed.
    """
    target = torch.zeros(vocab_size + 1, dtype=torch.float32)
    total = float(deck.total_copies) or 1.0
    for idx, count in deck.cards:
        target[idx] = count / total
    return target


def sample_partial(
    deck: Deck,
    rng: random.Random,
) -> tuple[list[int], int]:
    """Remove one card copy uniformly at random.

    "Uniformly at random" here means "weighted by multiplicity" — a
    card with 3 copies is three times more likely to have one of its
    copies removed than a card with 1 copy. This is what DESIGN.md
    calls "removing one card at a time uniformly at random" (uniform
    over the 60 *positions*, not the |unique| card ids).

    Returns ``(remaining_multiset, removed_card_index)`` so callers
    that need a per-mask one-hot target can build it without
    re-sampling. The ``remaining_multiset`` never contains PAD=0.
    """
    # Expand to the multiset — a 60-element list of ids.
    expanded: list[int] = []
    for idx, count in deck.cards:
        expanded.extend([idx] * count)
    if not expanded:
        return expanded, 0
    drop_at = rng.randrange(len(expanded))
    removed = expanded[drop_at]
    return expanded[:drop_at] + expanded[drop_at + 1 :], removed


@dataclass(frozen=True, slots=True)
class ProposalSample:
    """One training example: a partial-deck multiset + its label.

    ``partial_ids`` is a 1-D tensor of logical ids (variable length).
    The collate function right-pads a batch to the max length.
    """

    partial_ids: torch.Tensor  # (N,)  int64
    ink_multihot: torch.Tensor  # (6,) float32
    target_distribution: torch.Tensor  # (vocab_size + 1,) float32


class ProposalDataset(Dataset[ProposalSample]):
    """Turns a list of decks into an infinite stream of masked examples.

    ``samples_per_deck`` controls how many logical examples each deck
    contributes per epoch. Each ``__getitem__`` resamples a fresh mask
    so re-indexing the same deck in the same epoch produces different
    partials. The caller's RNG comes from ``torch.manual_seed`` for
    reproducibility; pass an explicit ``seed`` for tests that need to
    pin the exact sequence.

    ``target_mode`` selects between the two supported label formulations
    (see :class:`TargetMode`). FULL_DECK is cheaper (one precomputed
    target per deck) but gives the model the same label across all
    masks of a deck; ONE_HOT_REMOVED builds a per-sample one-hot on
    the card that was removed, which acts as label-level
    regularisation — helpful when the training set is small enough
    that the model would otherwise memorise the 782-deck target
    distribution directly.
    """

    def __init__(
        self,
        decks: list[Deck],
        *,
        vocab_size: int,
        samples_per_deck: int = 12,
        target_mode: TargetMode = TargetMode.FULL_DECK,
        seed: int | None = None,
    ) -> None:
        if not decks:
            raise ValueError("ProposalDataset requires at least one deck.")
        if samples_per_deck <= 0:
            raise ValueError("samples_per_deck must be positive.")
        self._decks = list(decks)
        self._vocab_size = vocab_size
        self._samples_per_deck = samples_per_deck
        self._target_mode = target_mode
        # Precompute the full-deck target once per deck. For ONE_HOT
        # mode it's unused at __getitem__ time but we keep the slot so
        # subclasses / eval paths can fall back to the marginal.
        self._full_deck_targets = tuple(target_distribution_from_deck(d, vocab_size) for d in decks)
        self._ink_vectors = tuple(ink_multihot(d.inks) for d in decks)
        self._seed = seed

    def __len__(self) -> int:
        return len(self._decks) * self._samples_per_deck

    @property
    def target_mode(self) -> TargetMode:
        return self._target_mode

    def __getitem__(self, index: int) -> ProposalSample:
        deck_index = index // self._samples_per_deck
        # ``random.Random`` per-call: combining the dataset seed with
        # the sample index makes the dataset deterministic when a seed
        # is set, without needing a shared mutable generator (which
        # would break multi-worker DataLoaders).
        rng = random.Random(None if self._seed is None else self._seed ^ index ^ 0x9E3779B1)
        deck = self._decks[deck_index]
        remaining, removed = sample_partial(deck, rng)
        if not remaining:
            # Defensive; a validated deck has ≥ 60 copies. If we ever
            # hit this, emit a single-PAD sequence so the batch shape
            # stays valid and the loss contribution is near-zero.
            partial_ids = torch.zeros(1, dtype=torch.long)
        else:
            partial_ids = torch.tensor(remaining, dtype=torch.long)

        if self._target_mode is TargetMode.ONE_HOT_REMOVED:
            # Per-mask one-hot on the removed card. Small cost: a
            # vocab-sized zero allocation per sample. Could be folded
            # into a smarter sparse target later if it shows up in
            # profiling — not worth optimising before it hurts.
            target = torch.zeros(self._vocab_size + 1, dtype=torch.float32)
            target[removed] = 1.0
        else:
            target = self._full_deck_targets[deck_index]

        return ProposalSample(
            partial_ids=partial_ids,
            ink_multihot=self._ink_vectors[deck_index],
            target_distribution=target,
        )


def collate_proposal(batch: list[ProposalSample]) -> dict[str, torch.Tensor]:
    """Right-pad variable-length partials and stack targets + ink vectors.

    Returns a dict with the three tensors the training step consumes:
    ``card_ids`` ``(B, N_max)`` int64, ``ink_multihot`` ``(B, 6)``
    float32, ``target_distribution`` ``(B, vocab_size + 1)`` float32.
    Padding uses 0 (PAD) so the key-padding mask in ProposalNet picks
    up the right positions.
    """
    if not batch:
        raise ValueError("collate_proposal received an empty batch.")
    max_len = max(sample.partial_ids.shape[0] for sample in batch)
    card_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, sample in enumerate(batch):
        n = sample.partial_ids.shape[0]
        card_ids[i, :n] = sample.partial_ids
    ink_multihot_batch = torch.stack([sample.ink_multihot for sample in batch], dim=0)
    target_batch = torch.stack([sample.target_distribution for sample in batch], dim=0)
    return {
        "card_ids": card_ids,
        "ink_multihot": ink_multihot_batch,
        "target_distribution": target_batch,
    }


def load_card_embeddings(path: Path) -> torch.Tensor:
    """Load ``card_embeddings.fp32.safetensors`` as a float32 tensor.

    Extracted as a separate helper so tests can substitute a tiny
    synthetic embedding table without a real encoder checkpoint. The
    safetensors format stores exactly one tensor for this file; we
    grab the first (and only) key defensively in case that convention
    ever changes.
    """
    from safetensors.torch import load_file  # lazy import; heavy wheel

    tensors: dict[str, torch.Tensor] = load_file(str(path))
    if not tensors:
        raise ValueError(f"{path} contains no tensors.")
    # Prefer an explicit key if present, otherwise fall back to the
    # first one (encoder export currently writes `embeddings`).
    key = next(iter(tensors))
    for candidate in ("embeddings", "card_embeddings"):
        if candidate in tensors:
            key = candidate
            break
    tensor = tensors[key].to(dtype=torch.float32)
    if tensor.dim() != 2:
        raise ValueError(
            f"{path}:{key} must be 2-D (rows, dim); got shape {tuple(tensor.shape)}",
        )
    return tensor


def load_vocab_size(vocab_path: Path) -> int:
    """Read ``vocab.json``'s ``size`` field (number of non-PAD entries)."""
    payload: dict[str, Any] = json.loads(vocab_path.read_text(encoding="utf8"))
    size = payload.get("size")
    if not isinstance(size, int) or size < 1:
        raise ValueError(f"{vocab_path}: missing or invalid 'size' field: {size!r}")
    return size
