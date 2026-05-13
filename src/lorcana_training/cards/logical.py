"""Collapse ``CardSet`` printings into logical cards.

In Lorcana the same card can be printed multiple times with identical
rules but different artwork/rarity — e.g. *Mickey Mouse - True Friend*
exists in 7 printings across the base set, promos, and enchanted
variants. For training we treat them as **one** card:

- The vocabulary has one entry per logical card (``name + version``).
- Max-copies rules apply to the logical card, not per printing — you
  can't run 4x base + 4x enchanted Mickey.
- Card text and structured features are taken from a single
  *canonical* printing so we don't have to arbitrate between cosmetic
  text differences (whitespace, smart quotes, reminder-text tweaks).

The collapse is deterministic given a pinned ``cards-vN`` so two runs
produce byte-identical outputs.

The canonical printing is picked as follows, intended to mean "latest
official printing":

1. Printings with numeric ``setCode`` (``"1"`` … ``"10"``) outrank
   alphabetic codes (``"P1"``, ``"C2"``, ``"D23"``). Numbered sets are
   the canonical release stream; promos/challenge sets frequently
   reprint older wording with typos we'd rather not adopt.
2. Within numeric printings, higher ``int(setCode)`` wins (newest set).
3. Tie-break on ``cardNumber`` (later number = later in the set order).
4. Tie-break on ``id`` as a deterministic fallback.

We also scan for groups where two printings disagree on gameplay-
affecting fields after whitespace normalisation. This is a soft
diagnostic — the list goes into ``dedup-report.json`` for review but
does not fail the run, because the 22% of groups that spread across
sets do so almost entirely for cosmetic reasons (see DESIGN.md).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..schemas.generated.card import Card
from ..schemas.generated.card_set import CardSet


_WHITESPACE_RE = re.compile(r"\s+")


def logical_key(card: Card) -> tuple[str, str]:
    """``(name, version-or-empty)``. The canonical identity of a card."""
    return (card.name, card.version or "")


def logical_id(card: Card | tuple[str, str]) -> str:
    """Stable string id for a logical card. Used as a dict key and a
    vocab identifier; the two-tuple form is for when you already have it.
    """
    name, version = logical_key(card) if isinstance(card, Card) else card
    return f"{name}|{version}"


def _setcode_rank(set_code: str, card_number: int, id_: str) -> tuple[int, int, str, int, str]:
    """Sort key: (numeric>alpha, int-value, setcode, cardNumber, id)."""
    try:
        numeric_val = int(set_code)
        return (1, numeric_val, set_code, card_number, id_)
    except ValueError:
        return (0, 0, set_code, card_number, id_)


def _functional_signature(card: Card) -> tuple[Any, ...]:
    """Whitespace-normalised fingerprint used to detect non-cosmetic
    drift between printings of the same logical card.

    Flavour text, card number, set code, and the image URL are
    deliberately excluded — those *must* differ across printings.
    """
    text = _WHITESPACE_RE.sub(" ", (card.text or "")).strip().lower()
    return (
        card.cost,
        tuple(sorted(i.value for i in card.inks)),
        tuple(sorted(t.value for t in card.types)),
        tuple(sorted(card.classifications or [])),
        text,
        card.lore,
        card.strength,
        card.willpower,
        card.inkwell,
        tuple(sorted(card.keywords or [])),
        card.move_cost,
    )


@dataclass(frozen=True, slots=True)
class LogicalCard:
    """One row of the vocabulary: a logical card and its printings."""

    logical_id: str
    name: str
    version: str  # empty string for non-character cards without a version
    canonical: Card
    printings: tuple[Card, ...]  # all printings including the canonical, in canonical-rank order

    @property
    def printing_ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.printings)


@dataclass(frozen=True, slots=True)
class DedupReport:
    total_printings: int
    total_logical: int
    groups_with_multiple_printings: int
    groups_with_functional_drift: int
    drift_samples: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class LogicalCardSet:
    cards: tuple[LogicalCard, ...]  # indexed positionally; the vocab builder assigns 1..N
    printing_to_logical_id: dict[str, str] = field(hash=False, compare=False)
    report: DedupReport


def build_logical_cards(card_set: CardSet, *, drift_sample_limit: int = 20) -> LogicalCardSet:
    """Group printings by ``(name, version)`` and pick a canonical.

    The returned ``cards`` tuple is sorted by ``logical_id`` so the
    ordering is deterministic and independent of the input order.
    """
    groups: dict[tuple[str, str], list[Card]] = defaultdict(list)
    for c in card_set.cards:
        groups[logical_key(c)].append(c)

    logical_cards: list[LogicalCard] = []
    printing_to_logical: dict[str, str] = {}
    drift: list[dict[str, Any]] = []
    multi_groups = 0

    for key in sorted(groups.keys()):
        name, version = key
        printings = sorted(
            groups[key],
            key=lambda c: _setcode_rank(c.set_code, c.card_number, c.id),
            reverse=True,
        )
        canonical = printings[0]
        lid = logical_id(key)

        if len(printings) > 1:
            multi_groups += 1
            signatures = {_functional_signature(c) for c in printings}
            if len(signatures) > 1 and len(drift) < drift_sample_limit:
                drift.append(
                    {
                        "logical_id": lid,
                        "printings": [
                            {
                                "id": c.id,
                                "setCode": c.set_code,
                                "cardNumber": c.card_number,
                            }
                            for c in printings
                        ],
                        "distinctSignatures": len(signatures),
                    }
                )

        logical_cards.append(
            LogicalCard(
                logical_id=lid,
                name=name,
                version=version,
                canonical=canonical,
                printings=tuple(printings),
            )
        )
        for p in printings:
            printing_to_logical[p.id] = lid

    report = DedupReport(
        total_printings=len(card_set.cards),
        total_logical=len(logical_cards),
        groups_with_multiple_printings=multi_groups,
        groups_with_functional_drift=sum(
            1 for c in logical_cards if len({_functional_signature(p) for p in c.printings}) > 1
        ),
        drift_samples=tuple(drift),
    )

    return LogicalCardSet(
        cards=tuple(logical_cards),
        printing_to_logical_id=printing_to_logical,
        report=report,
    )
