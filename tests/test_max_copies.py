"""Parity tests for `compute_max_copies` against shared fixtures.

The TypeScript side runs its own `computeMaxCopies` against the same
fixture file (`tests/fixtures/max_copies_cards.json`); CI fails if the
two implementations disagree on any case.
"""

from __future__ import annotations

import math

from lorcana_training.cards.max_copies import compute_max_copies


def test_default_cap() -> None:
    assert compute_max_copies({"text": ""}) == 4


def test_any_number() -> None:
    text = "You may have any number of cards named Dalmatian Puppy in your deck."
    assert compute_max_copies({"text": text}) == math.inf


def test_up_to() -> None:
    text = "You may have up to 6 copies of this card in your deck."
    assert compute_max_copies({"text": text}) == 6


def test_only() -> None:
    text = "You may only have 1 copy of this card in your deck."
    assert compute_max_copies({"text": text}) == 1
