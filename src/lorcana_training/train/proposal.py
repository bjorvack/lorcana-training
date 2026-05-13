"""Deprecated entry-point shim — real orchestration lives in :mod:`lorcana_training.proposal`.

The proposal-net stage grew its own subpackage (``proposal/run.py``,
``proposal/data.py``) because it needs more than one file and the
``train/`` directory originally held only single-purpose helpers
(maskers, samplers). This module re-exports the public API so any
``from lorcana_training.train.proposal import ...`` still resolves;
new code should import from ``lorcana_training.proposal`` directly.
"""

from ..proposal import ProposalOptions, ProposalResult, train_proposal

__all__ = ["ProposalOptions", "ProposalResult", "train_proposal"]
