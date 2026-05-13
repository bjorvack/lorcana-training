"""k-means over deck embeddings → ``archetype_centroids.json``.

Each training deck is represented as a single R^d vector formed by
summing its card embeddings weighted by copy count and L2-normalising.
We then run vanilla k-means (k = 20 by default, matching DESIGN.md
§Putting-it-together) and ship the resulting centroids.

Web-side usage: the novelty bonus for a candidate card is the
distance from the candidate's own embedding to the *nearest*
archetype centroid. Candidates that pull the deck toward an under-
represented region of the embedding space get a larger bonus.

Implementation notes:

- Pure-tensor k-means with k-means++ init so the result is
  deterministic given a fixed seed. We avoid ``sklearn`` to keep the
  dependency surface small (the training stack already has torch +
  numpy + onnx; adding sklearn for one clustering call is overkill).
- ``max_iters`` is set low (50) on purpose: centroids converge well
  under that for decks numbering in the thousands, and the per-epoch
  cost is trivial.
- Tiny/all-same clusters can degenerate. We re-seed any cluster that
  ends up empty with a random point from the dataset; DESIGN allows
  k < ``initial_k`` in pathological cases but 20 over ~2 800 decks
  is comfortable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ..proposal.data import Deck


DEFAULT_K = 20
_KMEANS_MAX_ITERS = 50
_KMEANS_TOL = 1e-4


@dataclass(frozen=True, slots=True)
class ArchetypeResult:
    centroids: torch.Tensor  # (k_effective, d) L2-normalised
    cluster_sizes: tuple[int, ...]
    iterations: int
    seed: int


def compute_deck_vectors(
    decks: list[Deck],
    card_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Return a ``(n_decks, embed_dim)`` tensor of L2-normalised
    weighted-sum deck vectors. Decks with zero total weight (the
    defensive fallback path in :class:`ProposalDataset`) fall back
    to a zero vector, which k-means discards as degenerate.
    """
    out = torch.zeros(len(decks), card_embeddings.shape[1], dtype=torch.float32)
    for i, deck in enumerate(decks):
        total = 0
        for card_id, count in deck.cards:
            out[i] += count * card_embeddings[card_id]
            total += count
        if total > 0:
            out[i] = out[i] / total
    return F.normalize(out, dim=-1)


def kmeans(
    vectors: torch.Tensor,
    *,
    k: int,
    seed: int = 0,
    max_iters: int = _KMEANS_MAX_ITERS,
) -> ArchetypeResult:
    """Run batched Euclidean k-means with k-means++ init."""
    if vectors.dim() != 2:
        raise ValueError(f"vectors must be (n, d); got {tuple(vectors.shape)}")
    n, d = vectors.shape
    if n == 0:
        raise ValueError("k-means needs at least one vector")
    k = min(k, n)

    generator = torch.Generator().manual_seed(seed)
    centroids = _kmeans_pp_init(vectors, k=k, generator=generator)

    prev_shift = float("inf")
    iterations = 0
    for iterations in range(1, max_iters + 1):
        # (n, k) squared-Euclidean distances.
        dists = torch.cdist(vectors, centroids, p=2) ** 2
        assignments = dists.argmin(dim=1)

        new_centroids = torch.zeros_like(centroids)
        cluster_sizes = torch.zeros(k, dtype=torch.long)
        for cluster in range(k):
            mask = assignments == cluster
            size = int(mask.sum())
            cluster_sizes[cluster] = size
            if size > 0:
                new_centroids[cluster] = vectors[mask].mean(dim=0)
            else:
                # Re-seed an empty cluster with the single vector
                # furthest from its current assignment. Keeps k
                # intact without a degenerate zero centroid.
                farthest = int(dists.min(dim=1).values.argmax())
                new_centroids[cluster] = vectors[farthest]

        shift = float((centroids - new_centroids).pow(2).sum().sqrt())
        centroids = new_centroids
        if abs(prev_shift - shift) < _KMEANS_TOL:
            break
        prev_shift = shift

    # L2-normalise so the web app can use cosine distance directly.
    centroids = F.normalize(centroids, dim=-1)
    return ArchetypeResult(
        centroids=centroids,
        cluster_sizes=tuple(int(x) for x in cluster_sizes.tolist()),
        iterations=iterations,
        seed=seed,
    )


def _kmeans_pp_init(
    vectors: torch.Tensor,
    *,
    k: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """k-means++ initialisation: pick the first centroid uniformly,
    then each subsequent one with probability proportional to its
    squared distance to the nearest already-chosen centroid.

    Standard enough that an off-by-one here would be noticed, but it
    matters enough for k-means convergence quality that rolling it
    by hand rather than inviting a sklearn dependency is worth it.
    """
    n = vectors.shape[0]
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    first = int(torch.randint(0, n, (1,), generator=generator).item())
    centroids = vectors[first].unsqueeze(0).clone()
    for _ in range(1, k):
        dists = torch.cdist(vectors, centroids, p=2).min(dim=1).values ** 2
        total = float(dists.sum())
        if total == 0.0:
            # All identical vectors — just pick any point.
            idx = int(torch.randint(0, n, (1,), generator=generator).item())
        else:
            probs = (dists / total).cpu()
            idx = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        centroids = torch.cat([centroids, vectors[idx].unsqueeze(0)], dim=0)
    return centroids


def write_archetype_centroids_json(
    result: ArchetypeResult,
    path: Path,
) -> None:
    """Serialise as JSON ``{ "dim": int, "k": int, "centroids": [[...]],
    "clusterSizes": [...], "iterations": int, "seed": int }``.

    Using JSON (not a binary format) keeps the web bundle
    self-describing — the archetype set changes rarely and is small
    enough that the bytes don't matter.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dim": int(result.centroids.shape[1]),
        "k": int(result.centroids.shape[0]),
        "centroids": result.centroids.tolist(),
        "clusterSizes": list(result.cluster_sizes),
        "iterations": result.iterations,
        "seed": result.seed,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")
