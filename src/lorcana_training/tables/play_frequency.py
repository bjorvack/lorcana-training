"""Per ``(ink_pair, card_id)`` play-frequency table.

For every ink-pair seen in the training decks, count how many times
each card appears across all decks with that ink-pair, then normalise
to a probability distribution.

Output schema (JSON, keyed by canonical sorted ink-pair string):

    {
      "amber|ruby": { "12": 0.042, "57": 0.028, ... },
      "emerald|steel": { ... },
      ...
      "_all": { ... }
    }

The ``_all`` row is the marginal across every deck regardless of ink
pair — the web app's fallback when the user's partial ink pair isn't
in the table (e.g. a new set's combo that hadn't shown up at training
time).

Canonical ink-pair string: sort the two ink names lexicographically
and join with ``"|"``. Mono-ink decks become ``"amber|amber"`` etc.
so the JSON schema stays uniform.

Floating-point precision: fractions are stored as doubles then
truncated to 6 decimal places on write. Truncation (rather than
rounding) keeps the sum ≤ 1.0 without a renormalisation pass.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from ..proposal.data import Deck


ALL_KEY = "_all"
_DECIMALS = 6


def _ink_pair_key(inks: tuple[str, ...]) -> str:
    return "|".join(sorted(i.lower() for i in inks))


def compute_play_frequency(decks: list[Deck]) -> dict[str, dict[int, float]]:
    """Raw frequency tables (no rounding)."""
    counts: dict[str, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))
    totals: dict[str, int] = defaultdict(int)

    for deck in decks:
        key = _ink_pair_key(deck.inks)
        for card_id, count in deck.cards:
            counts[key][card_id] += count
            counts[ALL_KEY][card_id] += count
            totals[key] += count
            totals[ALL_KEY] += count

    table: dict[str, dict[int, float]] = {}
    for ink_key, per_card in counts.items():
        denom = totals[ink_key] or 1
        table[ink_key] = {card_id: c / denom for card_id, c in per_card.items()}
    return table


def write_play_frequency_json(
    table: dict[str, dict[int, float]],
    path: Path,
) -> None:
    """Serialise the raw table with stable key ordering + fixed decimals.

    JSON dicts don't guarantee order; we sort both the outer
    ink-pair keys and the inner card-id keys so re-runs on the
    same data produce byte-identical files (makes sha256 manifest
    entries meaningful).
    """
    # Round *after* grouping — rounding per-card then summing back up
    # would drift off 1.0 for ink pairs with many cards.
    sorted_out: dict[str, dict[str, float]] = {}
    for ink_key in sorted(table.keys()):
        row = table[ink_key]
        sorted_out[ink_key] = {
            str(card_id): _truncate(p, _DECIMALS) for card_id, p in sorted(row.items()) if p > 0.0
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted_out, indent=2) + "\n", encoding="utf8")


def _truncate(value: float, decimals: int) -> float:
    # Truncate rather than round so row sums never exceed 1.0 after
    # the rounding pass. Web-side consumers comparing to e.g. 0.5 as
    # a threshold won't see a card tip over because of bankers'
    # rounding.
    factor = 10**decimals
    return int(value * factor) / factor
