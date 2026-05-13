"""Text normalisation used before tokenisation.

Three transformations, applied in order by :func:`normalise_card_text`:

1. **Reminder-text stripping.** Early Lorcana sets repeat the rules
   of every keyword in parens — e.g. ``Shift 3 (You may pay 3 {I}
   to play this on top of one of your characters named Elsa.)`` The
   reminder is 100 % redundant with the keyword itself, inconsistent
   across printings (some reprints drop or paraphrase it), and
   dominates the training signal for keyword-heavy cards. Verified
   manually across cards-v2026.05.13-01 that every parens span in
   ``Card.text`` is reminder text; there are no legitimate
   non-reminder parens in the rules-text field.

2. **Smart-quote normalisation.** Different printings of the same
   card sometimes drift between ``'`` and ``'`` / ``"`` and
   ``"``. We collapse the curly forms to their ASCII equivalents so
   two printings of *Dig a Little Deeper* encode to the same token
   sequence.

3. **Whitespace collapse.** A handful of cards carry stray double
   spaces in their source ("{E}, 1 {I} —  Look at..."). We collapse
   runs of whitespace to a single space and trim edges.

Flavor text lives on a separate ``flavor`` field and is never fed
to the encoder, so nothing there to worry about.
"""

from __future__ import annotations

import re


# Match a parens span greedily to the first matching ``)``. Reminder
# spans never nest in card text, so a simple "no closing paren inside"
# regex is correct.
_REMINDER_RE = re.compile(r"\s*\([^)]*\)")

# Collapse whitespace runs so we don't leave double spaces in the source
# or ugly trailing/leading space after stripping parens.
_WHITESPACE_RE = re.compile(r"\s+")

# Curly / smart quote to ASCII. Covers the four Unicode variants that
# show up in the current pool; extend this map if future sets
# introduce new ones (the existing unit tests will tell you).
_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",  # left single
        "\u2019": "'",  # right single
        "\u201c": '"',  # left double
        "\u201d": '"',  # right double
    }
)


def strip_reminder_text(text: str) -> str:
    """Remove parenthesised reminder text.

    Does *not* touch quotes or whitespace on its own beyond the
    cleanup needed to avoid leaving gaps where parens used to be.
    Separate entry point for callers that only want the reminders
    gone.
    """
    if not text:
        return text
    stripped = _REMINDER_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", stripped).strip()


def normalise_quotes(text: str) -> str:
    """Map curly quotes to their ASCII equivalents."""
    return text.translate(_QUOTE_MAP)


def normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and trim edges."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalise_card_text(text: str) -> str:
    """The single entry point every card-text consumer should use.

    Applies strip + quote-normalise + whitespace-collapse. Idempotent.
    """
    if not text:
        return text
    return normalise_whitespace(normalise_quotes(strip_reminder_text(text)))
