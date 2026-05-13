"""Validate a tournament ``Dataset`` against the pinned ``cards-vN`` +
rewrite every deck to logical indices.

Callers: :func:`validate_dataset` returns a :class:`ValidatedDataset`
containing the accepted :class:`LogicalDeck` s (with the metadata
needed for recency / stratification downstream) plus a detailed
:class:`ValidationReport`. The orchestration CLI in an upcoming
commit decides whether to abort based on the drop-rate policy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime

from ..cards.logical import LogicalCardSet
from ..cards.vocab import Vocab
from ..schemas.generated.dataset import Dataset
from ..schemas.generated.deck import Ink
from .legality import LegalityReport, is_tournament_legal
from .rewrite import LogicalDeck, rewrite_deck


@dataclass(frozen=True, slots=True)
class ValidLogicalDeck:
    """A validated deck after rewriting, with the metadata the
    downstream split/recency logic needs.
    """

    deck: LogicalDeck
    inks: tuple[Ink, ...]
    tournament_date: date
    tournament_url: str
    tournament_name: str
    placement: int | None
    player: str | None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    total_decks: int
    valid_decks: int
    dropped_unknown_printing: int
    dropped_illegal: int
    drop_reasons_histogram: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    @property
    def drop_rate(self) -> float:
        if self.total_decks == 0:
            return 0.0
        return (self.total_decks - self.valid_decks) / self.total_decks


@dataclass(frozen=True, slots=True)
class ValidatedDataset:
    decks: tuple[ValidLogicalDeck, ...]
    report: ValidationReport


def validate_dataset(
    dataset: Dataset,
    *,
    vocab: Vocab,
    logical_cards: LogicalCardSet,
    strict_legality: bool = False,
) -> ValidatedDataset:
    logical_index_by_id = vocab.build_index_map()
    accepted: list[ValidLogicalDeck] = []
    unknown_count = 0
    illegal_count = 0
    reason_counter: Counter[str] = Counter()
    total = 0

    for tournament in dataset.tournaments:
        # The generated schema keeps tournament.date as the raw ISO string
        # (regex-validated), so parse it into a real date here once.
        tdate = datetime.strptime(tournament.date, "%Y-%m-%d").date()
        turl = str(tournament.source_url)
        tname = tournament.name
        for entry in tournament.decks:
            total += 1
            rewrite = rewrite_deck(
                entry.deck,
                printing_to_logical_id=logical_cards.printing_to_logical_id,
                logical_index_by_id=logical_index_by_id,
            )
            if not rewrite.ok:
                unknown_count += 1
                reason_counter["unknown_printing_id"] += 1
                continue
            assert rewrite.deck is not None
            # `entry.deck.inks` is the Deck1 (inner) model's ink list
            # in the generated dataset schema; pass it through directly.
            report: LegalityReport = is_tournament_legal(
                entry.deck.inks,
                rewrite.deck,
                vocab,
                strict_legality=strict_legality,
            )
            if not report.ok:
                illegal_count += 1
                # Bucket reasons by their leading token so the histogram
                # is compact ("Deck has N cards..." -> "Deck has ...").
                for r in report.reasons:
                    reason_counter[_bucket_reason(r)] += 1
                continue
            accepted.append(
                ValidLogicalDeck(
                    deck=rewrite.deck,
                    inks=tuple(entry.deck.inks),
                    tournament_date=tdate,
                    tournament_url=turl,
                    tournament_name=tname,
                    placement=entry.placement,
                    player=entry.player,
                )
            )

    report = ValidationReport(
        total_decks=total,
        valid_decks=len(accepted),
        dropped_unknown_printing=unknown_count,
        dropped_illegal=illegal_count,
        drop_reasons_histogram=tuple(sorted(reason_counter.items(), key=lambda kv: -kv[1])),
    )
    return ValidatedDataset(decks=tuple(accepted), report=report)


def _bucket_reason(reason: str) -> str:
    """Collapse per-card reasons into broad categories for the histogram."""
    if reason.startswith("Deck has"):
        return "deck_size_below_60"
    if reason.startswith("Unknown logical index"):
        return "unknown_logical_index"
    if "exceeds cap" in reason:
        return "count_exceeds_max_copies"
    if "is not in deck inks" in reason:
        return "card_ink_not_in_deck_inks"
    if "is banned" in reason or "is not_legal" in reason:
        return "card_not_legal"
    return "other"
