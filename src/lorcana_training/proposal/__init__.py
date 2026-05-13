"""Train the proposal net (masked set completion over deck multisets)."""

from .data import (
    ProposalDataset,
    ProposalSample,
    collate_proposal,
    load_decks_jsonl,
)
from .run import ProposalOptions, ProposalResult, train_proposal

__all__ = [
    "ProposalDataset",
    "ProposalOptions",
    "ProposalResult",
    "ProposalSample",
    "collate_proposal",
    "load_decks_jsonl",
    "train_proposal",
]
