"""Tests for card-text normalisation (reminder-text stripping)."""

from __future__ import annotations

import pytest

from lorcana_training.text.normalise import (
    normalise_card_text,
    normalise_quotes,
    normalise_whitespace,
    strip_reminder_text,
)


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
    # Single callsite funnels every transformation so downstream
    # consumers (tokeniser + encoder) never disagree on normalisation.
    assert normalise_card_text("Shift 3 (reminder.)") == "Shift 3"


def test_normalise_quotes_collapses_smart_quotes() -> None:
    # The four curly variants that appear in the pool, ASCII-ified.
    assert normalise_quotes("your teammates\u2019 characters") == "your teammates' characters"
    assert normalise_quotes("\u2018hi\u2019") == "'hi'"
    assert normalise_quotes("\u201cok\u201d") == '"ok"'


def test_normalise_whitespace_collapses_runs() -> None:
    assert normalise_whitespace("a   b\t\tc") == "a b c"
    assert normalise_whitespace("  hi  ") == "hi"


def test_normalise_card_text_handles_smart_quotes_and_double_spaces() -> None:
    # Real-pool pattern: '{E}, 1 {I} —  Look at...' has a double space
    # after the em dash. And some reprints use curly apostrophes.
    raw = "{E}, 1 {I} \u2014  Look at \u2018top\u2019 card."
    expected = "{E}, 1 {I} \u2014 Look at 'top' card."
    assert normalise_card_text(raw) == expected


def test_normalise_card_text_composes_all_steps() -> None:
    # A stacked keyword card that mixes every pattern we normalise:
    # curly quote, reminder parens, and a double space.
    raw = "Shift 3 (You may pay 3 {I}.) \u2018Rush\u2019  means  the character can challenge."
    assert normalise_card_text(raw) == "Shift 3 'Rush' means the character can challenge."
