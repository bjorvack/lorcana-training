"""Deprecated shim — curriculum implementation lives in
:mod:`lorcana_training.evaluator.data`.

The phase enum + negative sampler were small enough to keep next to
the dataset that consumes them rather than spread across two files.
This module re-exports them for backwards compatibility."""

from ..evaluator.data import CurriculumPhase, NegativeSampler

__all__ = ["CurriculumPhase", "NegativeSampler"]
