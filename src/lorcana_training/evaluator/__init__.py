"""Train the per-step evaluator (curriculum-negative BCE discriminator)."""

from .data import (
    CardIndex,
    CurriculumPhase,
    EvaluatorDataset,
    NegativeSampler,
    build_card_index,
    collate_evaluator,
)
from .run import EvaluatorOptions, EvaluatorResult, train_evaluator

__all__ = [
    "CardIndex",
    "CurriculumPhase",
    "EvaluatorDataset",
    "EvaluatorOptions",
    "EvaluatorResult",
    "NegativeSampler",
    "build_card_index",
    "collate_evaluator",
    "train_evaluator",
]
