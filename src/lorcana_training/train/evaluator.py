"""Deprecated shim — real code lives in :mod:`lorcana_training.evaluator`.

The evaluator stage grew past what fits in a single file. See
``evaluator/run.py`` (orchestration) and ``evaluator/data.py``
(curriculum negatives). This module re-exports the public API so any
``from lorcana_training.train.evaluator import ...`` imports in older
code keep resolving.
"""

from ..evaluator import EvaluatorOptions, EvaluatorResult, train_evaluator

__all__ = ["EvaluatorOptions", "EvaluatorResult", "train_evaluator"]
