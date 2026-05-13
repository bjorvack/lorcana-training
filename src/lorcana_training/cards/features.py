"""Per-logical-card structured features for the encoder's structured head.

DESIGN.md defines the structured half of the card encoder's input:
cost one-hot, type multi-hot, ink multi-hot, classification multi-hot,
keyword multi-hot, plus lore/strength/willpower/moveCost normalised.
Inkwell is a single bool. The text half is learned separately during
``pretrain-encoder``.

We emit two artifacts under ``<prepared>/``:

- ``card_features.safetensors`` — an ``(N+1, D)`` float32 tensor. Row 0
  is all-zeros (PAD); rows 1..N line up 1:1 with the vocab indices.
- ``feature_schema.json`` — the exact layout (slices + class lists)
  so the encoder can read it without carrying its own duplicate copy.
  This is the single source of truth for "what does dim 37 mean?".

The feature list is derived from the actual card pool (classifications
+ keywords that actually appear in ``cards-vN``) rather than a
hand-maintained enum. That way a new set adding ``Resonate`` picks it
up automatically — there's a new keyword dim, everything else stays
the same. The feature_schema.json also records the set of classes at
build time so a retrained model can detect a mismatch when loaded
against a newer cards-vN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import save_file

from ..schemas.generated.card import Card
from .logical import LogicalCard
from .vocab import Vocab


# Upper bound for the cost one-hot. Costs above this fold into the last
# bin. The Lorcana pool has maxed out at 10-11 so far so this gives us
# room without blowing up dim.
_MAX_COST_BUCKET = 12

# Lorcana ink list in canonical order. The StrEnum in the generated
# schema uses the same ordering (enum.amber ... enum.steel).
_INKS = ("Amber", "Amethyst", "Emerald", "Ruby", "Sapphire", "Steel")

# Fixed type list; order is stable so tests can assert slice positions.
_TYPES = ("Character", "Action", "Song", "Item", "Location")


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """Layout of each row in the features tensor.

    Each ``*_slice`` is ``(start, length)``; ``dim = sum(lengths) + 4``
    (the four scalars: lore, strength, willpower, moveCost). Inkwell is
    a single dim appended at the end.
    """

    cost_slice: tuple[int, int]
    inks_slice: tuple[int, int]
    types_slice: tuple[int, int]
    classifications_slice: tuple[int, int]
    keywords_slice: tuple[int, int]
    lore_index: int
    strength_index: int
    willpower_index: int
    move_cost_index: int
    inkwell_index: int
    dim: int
    classifications: tuple[str, ...]
    keywords: tuple[str, ...]
    max_cost_bucket: int
    # Per-stat normalisers used in the forward path. Stored so
    # inference can rescale consistently.
    lore_max: int
    strength_max: int
    willpower_max: int
    move_cost_max: int

    def to_json(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "slices": {
                "cost": list(self.cost_slice),
                "inks": list(self.inks_slice),
                "types": list(self.types_slice),
                "classifications": list(self.classifications_slice),
                "keywords": list(self.keywords_slice),
            },
            "scalars": {
                "lore": self.lore_index,
                "strength": self.strength_index,
                "willpower": self.willpower_index,
                "moveCost": self.move_cost_index,
                "inkwell": self.inkwell_index,
            },
            "classes": {
                "inks": list(_INKS),
                "types": list(_TYPES),
                "classifications": list(self.classifications),
                "keywords": list(self.keywords),
            },
            "normalisers": {
                "maxCostBucket": self.max_cost_bucket,
                "loreMax": self.lore_max,
                "strengthMax": self.strength_max,
                "willpowerMax": self.willpower_max,
                "moveCostMax": self.move_cost_max,
            },
        }


def _discover_classes(
    logical_cards: tuple[LogicalCard, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Collect the classifications + keywords that actually appear."""
    classes: set[str] = set()
    kws: set[str] = set()
    for lc in logical_cards:
        if lc.canonical.classifications:
            classes.update(lc.canonical.classifications)
        if lc.canonical.keywords:
            kws.update(lc.canonical.keywords)
    return tuple(sorted(classes)), tuple(sorted(kws))


def _compute_normalisers(logical_cards: tuple[LogicalCard, ...]) -> tuple[int, int, int, int]:
    """Fit max lore/strength/willpower/moveCost from the pool. Falls
    back to 1 to keep the division well-defined for pools without any
    non-null values of a given stat (e.g. the first set without
    Locations)."""
    lore_max = 1
    strength_max = 1
    willpower_max = 1
    move_cost_max = 1
    for lc in logical_cards:
        c = lc.canonical
        if c.lore is not None:
            lore_max = max(lore_max, c.lore)
        if c.strength is not None:
            strength_max = max(strength_max, c.strength)
        if c.willpower is not None:
            willpower_max = max(willpower_max, c.willpower)
        if c.move_cost is not None:
            move_cost_max = max(move_cost_max, c.move_cost)
    return lore_max, strength_max, willpower_max, move_cost_max


def build_feature_schema(logical_cards: tuple[LogicalCard, ...]) -> FeatureSchema:
    classifications, keywords = _discover_classes(logical_cards)
    lore_max, strength_max, willpower_max, move_cost_max = _compute_normalisers(logical_cards)

    # Lay out the row: concatenation of one-hot/multi-hot blocks, then
    # scalars. Tracking the running cursor keeps the schema and the
    # encoder impossible to drift.
    cursor = 0
    cost_slice = (cursor, _MAX_COST_BUCKET)
    cursor += _MAX_COST_BUCKET
    inks_slice = (cursor, len(_INKS))
    cursor += len(_INKS)
    types_slice = (cursor, len(_TYPES))
    cursor += len(_TYPES)
    classifications_slice = (cursor, len(classifications))
    cursor += len(classifications)
    keywords_slice = (cursor, len(keywords))
    cursor += len(keywords)
    lore_index = cursor
    cursor += 1
    strength_index = cursor
    cursor += 1
    willpower_index = cursor
    cursor += 1
    move_cost_index = cursor
    cursor += 1
    inkwell_index = cursor
    cursor += 1

    return FeatureSchema(
        cost_slice=cost_slice,
        inks_slice=inks_slice,
        types_slice=types_slice,
        classifications_slice=classifications_slice,
        keywords_slice=keywords_slice,
        lore_index=lore_index,
        strength_index=strength_index,
        willpower_index=willpower_index,
        move_cost_index=move_cost_index,
        inkwell_index=inkwell_index,
        dim=cursor,
        classifications=classifications,
        keywords=keywords,
        max_cost_bucket=_MAX_COST_BUCKET,
        lore_max=lore_max,
        strength_max=strength_max,
        willpower_max=willpower_max,
        move_cost_max=move_cost_max,
    )


def _encode_card(
    card: Card, schema: FeatureSchema, *, class_idx: dict[str, int], keyword_idx: dict[str, int]
) -> np.ndarray:
    row = np.zeros(schema.dim, dtype=np.float32)
    # Cost one-hot (clipped).
    cost_bin = min(max(card.cost, 0), schema.max_cost_bucket - 1)
    row[schema.cost_slice[0] + cost_bin] = 1.0
    # Inks multi-hot.
    for ink in card.inks:
        row[schema.inks_slice[0] + _INKS.index(ink.value)] = 1.0
    # Types multi-hot.
    for ct in card.types:
        row[schema.types_slice[0] + _TYPES.index(ct.value)] = 1.0
    # Classifications multi-hot.
    for cls in card.classifications or []:
        idx = class_idx.get(cls)
        if idx is not None:
            row[schema.classifications_slice[0] + idx] = 1.0
    # Keywords multi-hot.
    for kw in card.keywords or []:
        idx = keyword_idx.get(kw)
        if idx is not None:
            row[schema.keywords_slice[0] + idx] = 1.0
    # Scalars (normalised to [0, 1]).
    if card.lore is not None:
        row[schema.lore_index] = card.lore / schema.lore_max
    if card.strength is not None:
        row[schema.strength_index] = card.strength / schema.strength_max
    if card.willpower is not None:
        row[schema.willpower_index] = card.willpower / schema.willpower_max
    if card.move_cost is not None:
        row[schema.move_cost_index] = card.move_cost / schema.move_cost_max
    # Inkwell flag.
    row[schema.inkwell_index] = 1.0 if card.inkwell else 0.0
    return row


def build_features(vocab: Vocab, schema: FeatureSchema) -> np.ndarray:
    """Return an ``(vocab.size + 1, schema.dim)`` float32 tensor.
    Row 0 is all-zeros for PAD; rows 1..N line up with vocab indices."""
    class_idx = {c: i for i, c in enumerate(schema.classifications)}
    keyword_idx = {k: i for i, k in enumerate(schema.keywords)}
    out = np.zeros((len(vocab.entries), schema.dim), dtype=np.float32)
    for i, entry in enumerate(vocab.entries):
        if entry is None:
            continue  # PAD row stays zero
        out[i] = _encode_card(entry.canonical, schema, class_idx=class_idx, keyword_idx=keyword_idx)
    return out


def write_features(
    features: np.ndarray, schema: FeatureSchema, *, out_dir: Path
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path = out_dir / "card_features.safetensors"
    save_file({"card_features": features}, str(feat_path))
    schema_path = out_dir / "feature_schema.json"
    schema_path.write_text(json.dumps(schema.to_json(), indent=2) + "\n", encoding="utf8")
    return {"card_features": feat_path, "feature_schema": schema_path}
