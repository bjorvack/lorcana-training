"""Build and serialise the training vocabulary.

The vocab is a stable bijection between integer indices (used
everywhere in the model as token ids) and logical cards. Index 0 is
``PAD``; indices 1..N are logical cards in ``logical_id`` order.

Two files get written to ``<prepared>/``:

- ``vocab.json`` — the primary artifact. Every index maps to the
  canonical printing's id plus the full set of printings that
  collapse into it.
- ``printing_to_logical.json`` — reverse lookup from any printing id
  (as they appear in tournament decks) to a logical index. The
  dataset loader uses this to rewrite printing ids into model ids
  before training.
- ``dedup-report.json`` — diagnostic: total counts + spot-check
  samples of groups where printings disagree on functional fields.

The vocab file carries a ``cardSetVersion`` (the sha256 hash of the
source ``cards.json``) so ``lorcana-web`` can refuse to load a model
whose vocab doesn't match its embedded cards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schemas.generated.card_set import CardSet
from .logical import DedupReport, LogicalCard, LogicalCardSet


PAD_INDEX = 0


@dataclass(frozen=True, slots=True)
class Vocab:
    pad_index: int
    # Parallel lists indexed 1..N. ``entries[0]`` is the PAD placeholder.
    entries: tuple[LogicalCard | None, ...]

    @property
    def size(self) -> int:
        """Number of real (non-PAD) entries."""
        return len(self.entries) - 1

    def index_of(self, logical_id: str) -> int:
        """Look up the index for a given logical id. O(n); for bulk
        use, keep the dict from ``build_index_map`` around instead.
        """
        for i, card in enumerate(self.entries):
            if card is not None and card.logical_id == logical_id:
                return i
        raise KeyError(logical_id)

    def build_index_map(self) -> dict[str, int]:
        return {c.logical_id: i for i, c in enumerate(self.entries) if c is not None}


def build_vocab(logical_cards: LogicalCardSet) -> Vocab:
    entries: list[LogicalCard | None] = [None]  # PAD at index 0
    entries.extend(logical_cards.cards)
    return Vocab(pad_index=PAD_INDEX, entries=tuple(entries))


def write_vocab(
    vocab: Vocab,
    logical_cards: LogicalCardSet,
    card_set: CardSet,
    *,
    out_dir: Path,
    cards_release_tag: str,
) -> dict[str, Path]:
    """Write vocab.json + printing_to_logical.json + dedup-report.json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the scraper's published cardSetVersion verbatim rather than
    # recomputing. That way `lorcana-web`'s manifest check is a string
    # comparison that can't silently disagree about normalisation rules.
    vocab_doc: dict[str, Any] = {
        "padIndex": vocab.pad_index,
        "size": vocab.size,
        "cardSetVersion": card_set.card_set_version,
        "cardsReleaseTag": cards_release_tag,
        "cards": [
            {
                "index": i,
                "logicalId": c.logical_id,
                "name": c.name,
                "version": c.version,
                "canonicalPrintingId": c.canonical.id,
                "canonicalSetCode": c.canonical.set_code,
                "canonicalCardNumber": c.canonical.card_number,
                "printingIds": list(c.printing_ids),
            }
            for i, c in enumerate(vocab.entries)
            if c is not None
        ],
    }
    vocab_path = out_dir / "vocab.json"
    vocab_path.write_text(json.dumps(vocab_doc, indent=2) + "\n", encoding="utf8")

    ptl_path = out_dir / "printing_to_logical.json"
    # Sort keys so git diffs are stable when a new cards-vN adds printings.
    ptl_path.write_text(
        json.dumps(dict(sorted(logical_cards.printing_to_logical_id.items())), indent=2) + "\n",
        encoding="utf8",
    )

    report_path = out_dir / "dedup-report.json"
    report_path.write_text(_report_to_json(logical_cards.report) + "\n", encoding="utf8")

    return {"vocab": vocab_path, "printing_to_logical": ptl_path, "dedup_report": report_path}


def _report_to_json(r: DedupReport) -> str:
    return json.dumps(
        {
            "totalPrintings": r.total_printings,
            "totalLogical": r.total_logical,
            "groupsWithMultiplePrintings": r.groups_with_multiple_printings,
            "groupsWithFunctionalDrift": r.groups_with_functional_drift,
            "driftSamples": list(r.drift_samples),
        },
        indent=2,
    )
