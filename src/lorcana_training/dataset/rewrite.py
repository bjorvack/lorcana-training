"""Rewrite tournament decks from printing ids to logical card indices.

The scraper publishes decks keyed by printing id (e.g. ``crd_407481…``
is a specific printing of *Mickey Mouse - True Friend*). For training
we operate on logical cards (see ``cards.logical``), so every deck
has to be rewritten before it can be fed to the models.

Counts collapse: a deck that lists 2 copies of printing A plus 2 of
printing B (both canonical to the same logical card) becomes a single
(logical_index, 4) entry. This is the point at which we discover
tournament decks that use multiple printings of the same card —
fairly common for e.g. enchanted + base variants — and cap them at
the logical level.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..schemas.generated.dataset import Deck


@dataclass(frozen=True, slots=True)
class LogicalDeck:
    """A deck after printing->logical rewriting.

    ``cards`` is a sorted tuple of ``(logical_index, count)`` pairs.
    Sorting is purely for deterministic serialisation; training code
    doesn't care about order.
    """

    cards: tuple[tuple[int, int], ...]

    @property
    def total_cards(self) -> int:
        return sum(count for _, count in self.cards)


@dataclass(frozen=True, slots=True)
class RewriteResult:
    """Outcome of rewriting a single deck."""

    deck: LogicalDeck | None
    unknown_printing_ids: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.deck is not None


def rewrite_deck(
    deck: Deck,
    *,
    printing_to_logical_id: dict[str, str],
    logical_index_by_id: dict[str, int],
) -> RewriteResult:
    """Rewrite a deck into logical card indices.

    An unknown printing id (e.g. a printing that was scraped before the
    currently pinned ``cards-vN`` contained it) is a fatal error for
    the individual deck — we return ``deck=None`` plus the list of
    unknown ids so the caller can decide whether to drop only that
    deck or abort the run.
    """
    counts: dict[int, int] = defaultdict(int)
    unknown: list[str] = []
    for c in deck.cards:
        logical_id = printing_to_logical_id.get(c.card_id)
        if logical_id is None:
            unknown.append(c.card_id)
            continue
        idx = logical_index_by_id.get(logical_id)
        if idx is None:
            unknown.append(c.card_id)
            continue
        counts[idx] += c.count

    if unknown:
        return RewriteResult(deck=None, unknown_printing_ids=tuple(unknown))
    sorted_pairs = tuple(sorted(counts.items()))
    return RewriteResult(deck=LogicalDeck(cards=sorted_pairs), unknown_printing_ids=())
