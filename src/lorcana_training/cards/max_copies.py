"""Mirror of the TypeScript ``computeMaxCopies`` in ``@bjorvack/lorcana-schemas``.

Both implementations are exercised against a shared fixture file
(``fixtures/max-copies-cards.json``) so they cannot drift in CI.
"""

from __future__ import annotations

import math
import re
from typing import Any

_ANY_NUMBER = re.compile(r"you may have any number of cards named", re.IGNORECASE)
_UP_TO = re.compile(r"you may have up to (\d+) copies of", re.IGNORECASE)
_ONLY = re.compile(r"you may only have (\d+) cop(?:y|ies) of", re.IGNORECASE)


def compute_max_copies(card: dict[str, Any]) -> float:
    text = card.get("text") or ""
    if _ANY_NUMBER.search(text):
        return math.inf
    m = _UP_TO.search(text)
    if m:
        return int(m.group(1))
    m = _ONLY.search(text)
    if m:
        return int(m.group(1))
    return 4
