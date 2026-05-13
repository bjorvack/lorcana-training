"""Build the per-card dataset consumed by the pretrain-encoder loop.

One row per logical card. Each row is a precomputed

    (token_ids (T,), attention_mask (T,), struct_features (D,))

tuple. The token sequence is left-padded to a fixed ``max_positions``
the caller picks up from :class:`CardEncoderConfig`.

The tokeniser is trained on the fly from the pool of non-empty card
texts + a derived list of reserved tokens (names + classifications +
keywords + glyphs). For our ~2 000-card corpus that takes well
under a second on CPU, so we don't bother caching it across runs.

The pretrain/heldout split is a deterministic hash of ``logical_id``
(so adding or removing cards doesn't shuffle every other card
between splits). Default heldout fraction ~10 %.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from safetensors.numpy import load_file
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from ..cards.features import FeatureSchema
from ..cards.logical import LogicalCardSet
from ..cards.vocab import Vocab
from ..text import (
    PAD_TOKEN,
    collect_reserved_tokens,
    normalise_card_text,
    train_tokeniser,
)


@dataclass(frozen=True, slots=True)
class PreparedPaths:
    vocab: Path
    card_features: Path
    feature_schema: Path


@dataclass(frozen=True, slots=True)
class PretrainData:
    vocab: Vocab
    logical_cards: LogicalCardSet
    schema: FeatureSchema
    features: np.ndarray  # (N+1, D) from prepare
    tokeniser: Tokenizer
    texts: list[str]  # indexed 0..N-1 (logical cards only; no PAD row)
    train_indices: list[int]
    heldout_indices: list[int]


def build_pretrain_tokeniser(
    logical_cards: LogicalCardSet,
    *,
    out_path: Path,
    vocab_size: int = 32_000,
) -> Tokenizer:
    """Train the BPE tokeniser over the pool of non-empty card texts.

    Card text is passed through :func:`normalise_card_text` first so
    parenthesised keyword reminders never enter the training corpus —
    see :mod:`lorcana_training.text.normalise` for why.
    """
    texts = [normalise_card_text(c.canonical.text) for c in logical_cards.cards if c.canonical.text]
    texts = [t for t in texts if t]  # drop anything that became empty after stripping
    reserved = collect_reserved_tokens(logical_cards.cards)
    return train_tokeniser(
        texts,
        out_path=out_path,
        vocab_size=vocab_size,
        reserved_tokens=reserved,
    )


def split_indices(
    logical_cards: LogicalCardSet,
    *,
    heldout_ratio: float = 0.10,
    seed: str = "pretrain-encoder",
) -> tuple[list[int], list[int]]:
    """Deterministic hash-based train/heldout split over logical cards.

    We hash ``(seed, logical_id)`` rather than using a random shuffle so
    adding a new card doesn't move every other card between splits.
    """
    train: list[int] = []
    heldout: list[int] = []
    boundary = int(heldout_ratio * (1 << 32))
    for i, card in enumerate(logical_cards.cards):
        h = hashlib.sha1(f"{seed}:{card.logical_id}".encode("utf8")).digest()
        # Use the first 4 bytes as a uniform int in [0, 2^32).
        bucket = int.from_bytes(h[:4], "big")
        if bucket < boundary:
            heldout.append(i)
        else:
            train.append(i)
    return train, heldout


def _load_features(card_features_path: Path) -> np.ndarray:
    tensors = load_file(str(card_features_path))
    return tensors["card_features"]


def build_pretrain_dataset(
    paths: PreparedPaths,
    *,
    logical_cards: LogicalCardSet,
    vocab: Vocab,
    schema: FeatureSchema,
    tokeniser: Tokenizer,
    heldout_ratio: float = 0.10,
) -> PretrainData:
    """Assemble the in-memory pretrain dataset from the artifacts that
    ``prepare`` already produced + a fitted tokeniser."""
    features = _load_features(paths.card_features)
    # Row 0 in features is PAD; logical cards occupy rows 1..N.
    if features.shape[0] != len(logical_cards.cards) + 1:
        raise ValueError(
            "features tensor row count does not match vocab size "
            f"(features={features.shape[0]}, cards={len(logical_cards.cards)})"
        )
    # Strip reminder-text parens here too so training + inference see
    # identical text (the encoder and the embedding export both read
    # from this list).
    texts = [normalise_card_text(c.canonical.text or "") for c in logical_cards.cards]
    train_idx, heldout_idx = split_indices(logical_cards, heldout_ratio=heldout_ratio)
    return PretrainData(
        vocab=vocab,
        logical_cards=logical_cards,
        schema=schema,
        features=features,
        tokeniser=tokeniser,
        texts=texts,
        train_indices=train_idx,
        heldout_indices=heldout_idx,
    )


class CardPretrainDataset(Dataset[dict[str, torch.Tensor]]):
    """Wraps :class:`PretrainData` for use with a torch DataLoader.

    Returns per-row dicts with ``token_ids`` (T,) and ``struct_features``
    (D,); batching + padding is handled by :func:`collate`.
    """

    def __init__(self, data: PretrainData, *, indices: Iterable[int], max_positions: int) -> None:
        self._data = data
        self._indices = list(indices)
        self._max_positions = max_positions
        self._pad_id = data.tokeniser.token_to_id(PAD_TOKEN)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        card_idx = self._indices[i]
        text = self._data.texts[card_idx]
        enc = self._data.tokeniser.encode(text)
        ids = enc.ids[: self._max_positions]
        token_ids = torch.full((self._max_positions,), self._pad_id, dtype=torch.long)
        token_ids[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        struct = torch.from_numpy(self._data.features[card_idx + 1].copy())
        return {"token_ids": token_ids, "struct_features": struct}


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack row dicts into a batch — each field is already fixed-shape."""
    return {
        "token_ids": torch.stack([row["token_ids"] for row in batch]),
        "struct_features": torch.stack([row["struct_features"] for row in batch]),
    }
