"""Python mirror of ``isTournamentLegal`` from ``@bjorvack/lorcana-schemas``.

This is the only place tournament rules live on the Python side, and
it runs against the same ``fixtures/max-copies-cards.json`` shared
fixture as the TypeScript implementation so the two can't drift (see
``tests/test_max_copies.py`` for ``compute_max_copies``; the tests
here extend that to the deck-level check).

Legality rules enforced:

1. Every card id in the deck exists in the vocab (and was rewritten
   to a logical index upstream in ``rewrite_deck``).
2. Every card is ``legality == 'legal'``.
3. For every card, ``count <= compute_max_copies(card)``.
4. Every ink the cards require is in the deck's declared ink pair.
5. Total card count is at least 60.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cards.max_copies import compute_max_copies
from ..cards.vocab import Vocab
from ..schemas.generated.card import Card
from ..schemas.generated.deck import Ink
from .rewrite import LogicalDeck


_MIN_DECK_SIZE = 60


@dataclass(frozen=True, slots=True)
class LegalityReport:
    ok: bool
    reasons: tuple[str, ...]


def is_tournament_legal(
    inks: list[Ink],
    logical_deck: LogicalDeck,
    vocab: Vocab,
    *,
    strict_legality: bool = False,
) -> LegalityReport:
    """Run the same five checks as the TypeScript ``isTournamentLegal``.

    ``inks`` is the declared ink pair from the source deck; ``logical_deck``
    is the same deck rewritten to logical indices (deduplicated on
    ``(name, version)``) — count caps are checked against the logical
    card, so 4x base + 4x enchanted Mickey is correctly flagged as 8
    copies of one logical card.

    ``strict_legality`` (default ``False``) controls the handling of the
    ``legality`` field on cards:

      - ``banned`` is always a hard failure (actually-banned cards that
        must not appear in any deck).
      - ``not_legal`` is tolerated by default because the scraped
        tournaments include decks played when older sets were rotated
        *in*; the current cards snapshot marks those sets as
        ``not_legal`` after rotation. Passing ``strict_legality=True``
        switches to current-format-only training data.
    """
    reasons: list[str] = []
    allowed_inks = {i.value for i in inks}
    total = 0

    for logical_index, count in logical_deck.cards:
        entry = vocab.entries[logical_index]
        if entry is None:
            reasons.append(f"Unknown logical index: {logical_index}")
            continue
        total += count
        card = entry.canonical

        legality = card.legality.value
        if legality == "banned" or (strict_legality and legality != "legal"):
            reasons.append(f"{card.name} is {legality}")

        cap = compute_max_copies(_card_as_dict(card))
        if count > cap:
            cap_str = "infinity" if cap == float("inf") else int(cap)
            reasons.append(f"{card.name}: {count} copies exceeds cap {cap_str}")

        for ink in card.inks:
            if ink.value not in allowed_inks:
                reasons.append(f"{card.name} requires {ink.value} which is not in deck inks")
                break

    if total < _MIN_DECK_SIZE:
        reasons.append(f"Deck has {total} cards; minimum is {_MIN_DECK_SIZE}")

    return LegalityReport(ok=not reasons, reasons=tuple(reasons))


def _card_as_dict(card: Card) -> dict[str, str]:
    """``compute_max_copies`` takes a dict with a ``text`` key. Adapt."""
    return {"text": card.text or ""}
