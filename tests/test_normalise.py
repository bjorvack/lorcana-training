"""Tests for card-text normalisation (reminder-text stripping)."""

from __future__ import annotations

import pytest

from lorcana_training.text.normalise import normalise_card_text, strip_reminder_text


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "Shift 3 (You may pay 3 {I} to play this on top of one of your characters named Elsa.)",
            "Shift 3",
        ),
        (
            "Rush (This character can challenge the turn they're played.)",
            "Rush",
        ),
        (
            "Resist +1 (Damage dealt to them is reduced by 1.)",
            "Resist +1",
        ),
        # Multiple keywords stacked — each reminder stripped independently.
        (
            "Evasive (Only characters with Evasive can challenge this character.) Ward (Opponents can't choose this character except to challenge.)",
            "Evasive Ward",
        ),
        # Rules text around a reminder block stays intact, spacing collapses.
        (
            "Your characters gain Rush (They can challenge the turn they're played.) this turn.",
            "Your characters gain Rush this turn.",
        ),
        ("", ""),
        ("No parens here.", "No parens here."),
    ],
)
def test_strip_reminder_text(raw: str, expected: str) -> None:
    assert strip_reminder_text(raw) == expected


def test_strip_reminder_text_is_idempotent() -> None:
    once = strip_reminder_text("Shift 3 (reminder here)")
    twice = strip_reminder_text(once)
    assert once == twice


def test_normalise_card_text_is_the_entry_point() -> None:
    # Today normalise_card_text == strip_reminder_text; documenting the
    # contract so future additions to normalise() stay funnelled
    # through this single callsite.
    assert normalise_card_text("Shift 3 (reminder.)") == "Shift 3"
