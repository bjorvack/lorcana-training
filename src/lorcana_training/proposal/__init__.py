"""Train the proposal net (masked set completion over deck multisets)."""

from .data import (
    ProposalDataset,
    ProposalSample,
    TargetMode,
    collate_proposal,
    load_decks_jsonl,
)
from .run import ProposalOptions, ProposalResult, train_proposal

__all__ = [
    "ProposalDataset",
    "ProposalOptions",
    "ProposalResult",
    "ProposalSample",
    "TargetMode",
    "collate_proposal",
    "load_decks_jsonl",
    "train_proposal",
]
