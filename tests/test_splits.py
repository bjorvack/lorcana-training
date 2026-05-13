"""Tests for build_splits / write_splits."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from lorcana_training.dataset.rewrite import LogicalDeck
from lorcana_training.dataset.splits import (
    build_splits,
    subtract_months,
    write_splits,
)
from lorcana_training.dataset.validate import ValidatedDataset, ValidationReport, ValidLogicalDeck
from lorcana_training.schemas.generated.deck import Ink


def _dk(
    *,
    cards: tuple[tuple[int, int], ...] = ((1, 60),),
    inks: tuple[str, ...] = ("Amber", "Steel"),
    d: date = date(2025, 6, 1),
    placement: int | None = 1,
) -> ValidLogicalDeck:
    return ValidLogicalDeck(
        deck=LogicalDeck(cards=cards),
        inks=tuple(Ink(i) for i in inks),
        tournament_date=d,
        tournament_url="https://example.test/t1",
        tournament_name="T1",
        placement=placement,
        player="p",
    )


def _ds(*decks: ValidLogicalDeck) -> ValidatedDataset:
    return ValidatedDataset(
        decks=decks,
        report=ValidationReport(
            total_decks=len(decks),
            valid_decks=len(decks),
            dropped_unknown_printing=0,
            dropped_illegal=0,
            drop_reasons_histogram=(),
        ),
    )


def test_subtract_months_handles_month_and_year_wrap() -> None:
    assert subtract_months(date(2026, 5, 13), 12) == date(2025, 5, 13)
    assert subtract_months(date(2025, 3, 15), 6) == date(2024, 9, 15)
    # Day clamp on shorter months.
    assert subtract_months(date(2024, 3, 31), 1) == date(2024, 2, 29)  # leap year
    assert subtract_months(date(2025, 3, 31), 1) == date(2025, 2, 28)


def test_reference_date_is_latest_tournament() -> None:
    decks = [
        _dk(d=date(2024, 1, 1)),
        _dk(d=date(2025, 6, 1)),
        _dk(d=date(2024, 12, 31)),
    ]
    splits = build_splits(_ds(*decks), proposal_recency_months=12)
    assert splits.report.reference_date == date(2025, 6, 1)
    assert splits.report.proposal_cutoff_date == date(2024, 6, 1)


def test_proposal_cutoff_filters_older_decks() -> None:
    # 3 decks within the last 12 months + 3 older.
    decks = [
        _dk(d=date(2025, 6, 1)),
        _dk(d=date(2025, 3, 1)),
        _dk(d=date(2024, 8, 1)),  # inside window
        _dk(d=date(2024, 5, 1)),  # outside window
        _dk(d=date(2023, 12, 1)),  # outside
        _dk(d=date(2023, 1, 1)),  # outside
    ]
    splits = build_splits(_ds(*decks), proposal_recency_months=12, heldout_ratio=0.0, seed=0)
    # Small strata collapse to train_all; no held-out. Proposal = within window.
    assert splits.report.heldout == 0
    assert splits.report.train_evaluator == 6
    assert splits.report.train_proposal == 3  # decks from 2024-06 onward


def test_stratified_heldout_samples_across_strata() -> None:
    # 4 strata x 10 decks each. Each stratum should contribute ~1 heldout.
    decks: list[ValidLogicalDeck] = []
    for i in range(10):
        decks.append(_dk(d=date(2025, 6, 1), inks=("Amber", "Steel")))
        decks.append(_dk(d=date(2025, 6, 1), inks=("Amber", "Ruby")))
        decks.append(_dk(d=date(2025, 7, 1), inks=("Amber", "Steel")))
        decks.append(_dk(d=date(2025, 7, 1), inks=("Amber", "Ruby")))
    splits = build_splits(_ds(*decks), proposal_recency_months=12, heldout_ratio=0.10, seed=42)
    assert splits.report.strata_count == 4
    # With 10 per stratum at 10% ratio: exactly 1 heldout each.
    assert splits.report.heldout == 4
    assert splits.report.train_evaluator == 36
    # train_proposal is evaluator filtered by recency; all decks here are
    # within 12 months of the reference so they match.
    assert splits.report.train_proposal == 36


def test_small_strata_go_entirely_to_train() -> None:
    # Single deck in its own (ink, month) stratum.
    splits = build_splits(_ds(_dk()), proposal_recency_months=12)
    assert splits.report.heldout == 0
    assert splits.report.singleton_strata == 1
    assert splits.report.train_evaluator == 1


def test_deterministic_seed_reproduces_split() -> None:
    decks = [_dk(d=date(2025, 6, 1), placement=p) for p in range(1, 21)]
    ds = _ds(*decks)
    a = build_splits(ds, heldout_ratio=0.10, seed=7)
    b = build_splits(ds, heldout_ratio=0.10, seed=7)
    assert [d.placement for d in a.heldout] == [d.placement for d in b.heldout]
    # A different seed shuffles differently.
    c = build_splits(ds, heldout_ratio=0.10, seed=8)
    assert (
        [d.placement for d in a.heldout] != [d.placement for d in c.heldout]
        or len(a.heldout) != len(c.heldout)
    )


def test_write_splits_round_trips(tmp_path: Path) -> None:
    decks = [_dk(d=date(2025, 6, 1), placement=p) for p in range(1, 11)]
    splits = build_splits(_ds(*decks), heldout_ratio=0.20, seed=0)
    paths = write_splits(splits, out_dir=tmp_path)
    assert paths["train_evaluator"].exists()
    lines = paths["train_evaluator"].read_text().strip().splitlines()
    assert len(lines) == splits.report.train_evaluator
    row = json.loads(lines[0])
    # Shape check.
    assert set(row.keys()) == {"cards", "inks", "tournament", "placement", "player"}
    report = json.loads(paths["splits_report"].read_text())
    assert report["total"] == 10
    assert report["heldoutRatio"] == 0.20
