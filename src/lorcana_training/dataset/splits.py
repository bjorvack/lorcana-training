"""Per-model splits + recency filter.

Consumes a :class:`ValidatedDataset` and produces three deck lists:

- ``train_proposal`` — decks within the proposal's recency window
  (trailing N months relative to the latest tournament in the dataset,
  default 12 per DESIGN.md).
- ``train_evaluator`` — every validated deck *except* the held-out
  slice.
- ``heldout`` — a ~10% slice pulled off the full validated set,
  stratified by ``(ink_pair, year_month)`` so months and matchups
  both stay representative in the held-out eval.

Each split is written as JSON Lines (one deck per line) so the
training stages can stream them without loading the whole thing
into memory. JSONL was chosen over parquet at this stage because
(a) the current dataset is ~1k decks and the parquet overhead
(pyarrow is a 70 MB wheel) isn't justified, and (b) JSONL is
trivially inspectable from the shell with ``jq``. The loader API
treats the format as an implementation detail so parquet can
replace it later with no call-site changes.
"""

from __future__ import annotations

import calendar
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .validate import ValidatedDataset, ValidLogicalDeck


@dataclass(frozen=True, slots=True)
class SplitReport:
    total: int
    train_proposal: int
    train_evaluator: int
    heldout: int
    proposal_cutoff_date: date | None
    reference_date: date
    heldout_ratio: float
    strata_count: int
    singleton_strata: int  # strata with one deck; go entirely into train


@dataclass(frozen=True, slots=True)
class Splits:
    train_proposal: tuple[ValidLogicalDeck, ...]
    train_evaluator: tuple[ValidLogicalDeck, ...]
    heldout: tuple[ValidLogicalDeck, ...]
    report: SplitReport


def subtract_months(d: date, months: int) -> date:
    """Calendar-month subtraction. ``date(2026, 2, 29) - 12`` → ``date(2025, 2, 28)``."""
    year = d.year
    m = d.month - months
    while m <= 0:
        m += 12
        year -= 1
    last_day_in_month = calendar.monthrange(year, m)[1]
    return date(year, m, min(d.day, last_day_in_month))


def _stratum_key(deck: ValidLogicalDeck) -> tuple[tuple[str, ...], tuple[int, int]]:
    inks = tuple(sorted(i.value for i in deck.inks))
    d = deck.tournament_date
    return inks, (d.year, d.month)


def build_splits(
    validated: ValidatedDataset,
    *,
    proposal_recency_months: int = 12,
    heldout_ratio: float = 0.10,
    seed: int = 0,
) -> Splits:
    if not validated.decks:
        return Splits(
            train_proposal=(),
            train_evaluator=(),
            heldout=(),
            report=SplitReport(
                total=0,
                train_proposal=0,
                train_evaluator=0,
                heldout=0,
                proposal_cutoff_date=None,
                reference_date=date.min,
                heldout_ratio=heldout_ratio,
                strata_count=0,
                singleton_strata=0,
            ),
        )

    # Reference = latest tournament in the dataset. Makes runs reproducible:
    # the same validated dataset always produces the same cutoff, regardless
    # of when the script was invoked.
    reference_date = max(d.tournament_date for d in validated.decks)
    proposal_cutoff = subtract_months(reference_date, proposal_recency_months)

    # Stratify deterministically. random.Random with a seed is enough:
    # we don't need crypto randomness, we just need reruns of the same
    # (config, validated dataset) to emit the same split.
    rng = random.Random(seed)
    strata: dict[tuple[tuple[str, ...], tuple[int, int]], list[ValidLogicalDeck]] = defaultdict(list)
    for deck in validated.decks:
        strata[_stratum_key(deck)].append(deck)

    heldout: list[ValidLogicalDeck] = []
    train_all: list[ValidLogicalDeck] = []
    singletons = 0
    for key, decks in strata.items():
        shuffled = list(decks)
        rng.shuffle(shuffled)
        # Minimum stratum size for contributing a held-out deck. Under 10
        # decks we'd rather keep everything in train than skew the heldout
        # toward a tiny (ink, month) combo.
        if len(shuffled) < 10:
            singletons += 1 if len(shuffled) == 1 else 0
            train_all.extend(shuffled)
            continue
        n_heldout = max(1, int(round(len(shuffled) * heldout_ratio)))
        heldout.extend(shuffled[:n_heldout])
        train_all.extend(shuffled[n_heldout:])

    train_proposal = [d for d in train_all if d.tournament_date >= proposal_cutoff]

    report = SplitReport(
        total=len(validated.decks),
        train_proposal=len(train_proposal),
        train_evaluator=len(train_all),
        heldout=len(heldout),
        proposal_cutoff_date=proposal_cutoff,
        reference_date=reference_date,
        heldout_ratio=heldout_ratio,
        strata_count=len(strata),
        singleton_strata=singletons,
    )
    return Splits(
        train_proposal=tuple(train_proposal),
        train_evaluator=tuple(train_all),
        heldout=tuple(heldout),
        report=report,
    )


def _deck_to_json(deck: ValidLogicalDeck) -> dict[str, Any]:
    return {
        "cards": [[idx, count] for idx, count in deck.deck.cards],
        "inks": [i.value for i in deck.inks],
        "tournament": {
            "url": deck.tournament_url,
            "name": deck.tournament_name,
            "date": deck.tournament_date.isoformat(),
        },
        "placement": deck.placement,
        "player": deck.player,
    }


def _write_jsonl(decks: tuple[ValidLogicalDeck, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        for deck in decks:
            f.write(json.dumps(_deck_to_json(deck), separators=(",", ":")) + "\n")


def write_splits(splits: Splits, *, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "train_proposal": out_dir / "train.proposal.jsonl",
        "train_evaluator": out_dir / "train.evaluator.jsonl",
        "heldout": out_dir / "heldout.jsonl",
    }
    _write_jsonl(splits.train_proposal, paths["train_proposal"])
    _write_jsonl(splits.train_evaluator, paths["train_evaluator"])
    _write_jsonl(splits.heldout, paths["heldout"])

    report_path = out_dir / "splits-report.json"
    r = splits.report
    report_path.write_text(
        json.dumps(
            {
                "total": r.total,
                "trainProposal": r.train_proposal,
                "trainEvaluator": r.train_evaluator,
                "heldout": r.heldout,
                "proposalCutoffDate": r.proposal_cutoff_date.isoformat() if r.proposal_cutoff_date else None,
                "referenceDate": r.reference_date.isoformat() if r.reference_date != date.min else None,
                "heldoutRatio": r.heldout_ratio,
                "strataCount": r.strata_count,
                "singletonStrata": r.singleton_strata,
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    paths["splits_report"] = report_path
    return paths
