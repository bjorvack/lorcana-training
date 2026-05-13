"""Empirical tables shipped alongside the ONNX models.

Two deterministic derivations of the training dataset the web-side
search loop needs in addition to the neural nets:

- ``play_frequency.json`` — per ``(ink_pair, card_id)`` frequency the
  card appears across real decks. Used for the meta-closeness penalty
  inside the search's scoring formula.
- ``archetype_centroids.json`` — a small k-means over deck embeddings
  (sum of card embeddings, L2-normalised). Used for the novelty-bonus
  term: a card whose embedding pulls a partial deck away from every
  known archetype centroid gets a novelty boost.

Both tables are computed once at release time and shipped as JSON in
the ``model-vN`` bundle. See ``tables/run.py`` for the orchestration
and per-table module docstrings for the math.
"""

from .run import (
    BuildTablesOptions,
    BuildTablesResult,
    build_tables,
)

__all__ = [
    "BuildTablesOptions",
    "BuildTablesResult",
    "build_tables",
]
