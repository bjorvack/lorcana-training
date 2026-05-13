"""Text normalisation used before tokenisation.

The only transformation right now is stripping parenthesised
*reminder text* from card bodies. In early Lorcana sets, every
keyword printing repeats its rules in parens — e.g.

    Shift 3 (You may pay 3 {I} to play this on top of one of your
    characters named Elsa.) Rush (This character can challenge the
    turn they're played.)

The reminder text:

1. Is 100% redundant with the keyword itself — the keyword is the
   authoritative name of the mechanic and already appears in the
   card text.
2. Dominates the training signal. A card with two keywords becomes
   mostly reminder prose; the model ends up trying to *reproduce*
   the reminders rather than learn what the keywords *do* from
   their distribution across the pool.
3. Is inconsistently present across printings — some reprints drop
   the reminder, some keep it, some paraphrase it slightly.

Stripping is a clean universal rule: every parens-wrapped segment in
``Card.text`` in the current pool is reminder text (verified
manually over cards-v2026.05.13-01). There are no legitimate
non-reminder parens in the rules-text field.

Flavor text lives on a separate ``flavor`` field and is never fed
to the encoder, so nothing there to worry about.
"""

from __future__ import annotations

import re


# Match a parens span greedily to the first matching ``)``. Reminder
# spans never nest in card text, so a simple "no closing paren inside"
# regex is correct.
_REMINDER_RE = re.compile(r"\s*\([^)]*\)")

# Collapse whitespace after removing spans so we don't leave "Shift   Rush"
# or trailing spaces behind.
_WHITESPACE_RE = re.compile(r"\s+")


def strip_reminder_text(text: str) -> str:
    """Return ``text`` with parenthesised reminder text removed.

    Idempotent: running twice returns the same result as once.
    """
    if not text:
        return text
    stripped = _REMINDER_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def normalise_card_text(text: str) -> str:
    """Entry point used everywhere we feed card text to the encoder.

    Thin wrapper today (only strips reminders); kept as a stable
    name so future normalisations — whitespace clean-up, smart-quote
    canonicalisation, keyword-case normalisation — have one obvious
    place to land.
    """
    return strip_reminder_text(text)
