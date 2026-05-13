"""Masking utilities for pretrain-encoder.

Two kinds of masking:

- :func:`mask_tokens` — BERT-style MLM masking. Picks 15% of non-
  special positions; of those, 80% get replaced with [MASK], 10% with
  a random token, 10% left unchanged (but still scored in the loss).
  Labels are -100 everywhere except the 15% so cross-entropy ignores
  the unmasked positions.

- :func:`mask_structured_blocks` — denoising-AE masking over the
  structured features tensor. With probability ``block_drop_prob``,
  zeros out one of the feature schema's blocks (``cost`` / ``inks``
  / ``types`` / ``classifications`` / ``keywords`` / the scalar stats
  group). Blockwise rather than per-dim masking because the blocks
  are semantically coherent — scrambling "half the ink dims" doesn't
  simulate a realistic "we don't know the card's inks" situation.

Both return a copy of the input; they never mutate in place, so the
training loop can keep the originals around for the loss targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..cards.features import FeatureSchema
from ..models.pretrain_heads import MLM_IGNORE_INDEX


@dataclass(frozen=True, slots=True)
class TokenMaskResult:
    input_ids: torch.Tensor
    labels: torch.Tensor  # MLM_IGNORE_INDEX outside of masked positions


def mask_tokens(
    token_ids: torch.Tensor,
    *,
    mask_token_id: int,
    vocab_size: int,
    special_token_ids: set[int],
    mask_prob: float = 0.15,
    replace_with_mask_prob: float = 0.8,
    replace_with_random_prob: float = 0.1,
    generator: torch.Generator | None = None,
) -> TokenMaskResult:
    """BERT-style MLM masking. See module docstring."""
    if token_ids.dim() != 2:
        raise ValueError(f"expected (B, T); got {tuple(token_ids.shape)}")
    if not (0.0 < mask_prob <= 1.0):
        raise ValueError(f"mask_prob must be in (0, 1]; got {mask_prob}")

    device = token_ids.device
    special_ids_tensor = torch.tensor(sorted(special_token_ids), device=device)
    # True where the position is eligible for masking (i.e. not a special token).
    eligible = ~torch.isin(token_ids, special_ids_tensor)
    mask_draw = _uniform_like(token_ids, generator) < mask_prob
    masked_positions = mask_draw & eligible

    labels = torch.full_like(token_ids, MLM_IGNORE_INDEX)
    labels[masked_positions] = token_ids[masked_positions]

    input_ids = token_ids.clone()
    # Of the masked positions, decide each row's fate:
    #   - with `replace_with_mask_prob`:   -> [MASK]
    #   - with `replace_with_random_prob`: -> random non-special token
    #   - otherwise:                        -> unchanged (still scored)
    action_draw = _uniform_like(token_ids, generator)
    with_mask = masked_positions & (action_draw < replace_with_mask_prob)
    with_random = (
        masked_positions
        & ~with_mask
        & (action_draw < replace_with_mask_prob + replace_with_random_prob)
    )
    input_ids[with_mask] = mask_token_id
    if with_random.any():
        random_tokens = torch.randint(
            0, vocab_size, size=(int(with_random.sum().item()),), device=device, generator=generator
        )
        input_ids[with_random] = random_tokens

    return TokenMaskResult(input_ids=input_ids, labels=labels)


@dataclass(frozen=True, slots=True)
class StructMaskResult:
    features: torch.Tensor  # masked copy
    dropped_blocks: list[str]  # names of blocks zeroed, per batch row — diagnostic


def mask_structured_blocks(
    features: torch.Tensor,
    *,
    schema: FeatureSchema,
    block_drop_prob: float = 0.30,
    generator: torch.Generator | None = None,
) -> StructMaskResult:
    """Randomly zero out whole feature blocks for each row independently.

    The six candidate blocks are the five multi-hot slices (cost, inks,
    types, classifications, keywords) plus a grouped "scalars" block
    that bundles lore/strength/willpower/moveCost/inkwell. Grouping
    scalars keeps the denoising signal coarse — randomly zeroing one
    stat at a time would be trivially reconstructible from the others.
    """
    if features.dim() != 2:
        raise ValueError(f"expected (B, D); got {tuple(features.shape)}")
    if not (0.0 <= block_drop_prob <= 1.0):
        raise ValueError(f"block_drop_prob must be in [0, 1]; got {block_drop_prob}")

    out = features.clone()
    blocks: list[tuple[str, list[int]]] = [
        ("cost", list(range(schema.cost_slice[0], schema.cost_slice[0] + schema.cost_slice[1]))),
        ("inks", list(range(schema.inks_slice[0], schema.inks_slice[0] + schema.inks_slice[1]))),
        ("types", list(range(schema.types_slice[0], schema.types_slice[0] + schema.types_slice[1]))),
        (
            "classifications",
            list(
                range(
                    schema.classifications_slice[0],
                    schema.classifications_slice[0] + schema.classifications_slice[1],
                )
            ),
        ),
        (
            "keywords",
            list(range(schema.keywords_slice[0], schema.keywords_slice[0] + schema.keywords_slice[1])),
        ),
        (
            "scalars",
            [
                schema.lore_index,
                schema.strength_index,
                schema.willpower_index,
                schema.move_cost_index,
                schema.inkwell_index,
            ],
        ),
    ]

    rng = torch.rand(features.size(0), len(blocks), generator=generator, device=features.device)
    dropped_per_row: list[list[str]] = [[] for _ in range(features.size(0))]
    for b_idx, (name, indices) in enumerate(blocks):
        drop_mask = rng[:, b_idx] < block_drop_prob  # (B,)
        if drop_mask.any():
            idx_tensor = torch.tensor(indices, device=features.device, dtype=torch.long)
            rows = drop_mask.nonzero(as_tuple=False).squeeze(1)
            out[rows.unsqueeze(1), idx_tensor.unsqueeze(0)] = 0.0
            for r in rows.tolist():
                dropped_per_row[r].append(name)

    # Flatten diagnostic into a single list with row index prefix.
    flat = [f"{i}:{name}" for i, names in enumerate(dropped_per_row) for name in names]
    return StructMaskResult(features=out, dropped_blocks=flat)


def _uniform_like(x: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    return torch.rand(x.shape, device=x.device, generator=generator)
