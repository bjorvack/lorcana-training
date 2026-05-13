"""Top-level orchestration for ``lorcana-train build-tables``.

Builds both ``play_frequency.json`` and ``archetype_centroids.json``
from the prepared training set + the encoder's card embeddings, plus
a small ``tables-manifest.json`` recording provenance + file sha256s.

Inputs mirror every other post-prepare stage: ``prepared/manifest.json``,
``encoder-export/encoder-manifest.json``, and the evaluator's training
split for the deck source (full set, no recency filter).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT
from ..proposal.data import load_card_embeddings, load_decks_jsonl, load_vocab_size
from .archetype_centroids import (
    DEFAULT_K,
    compute_deck_vectors,
    kmeans,
    write_archetype_centroids_json,
)
from .play_frequency import compute_play_frequency, write_play_frequency_json


@dataclass(frozen=True, slots=True)
class BuildTablesOptions:
    prepared_dir: Path = REPO_ROOT / "prepared"
    encoder_export_dir: Path = REPO_ROOT / "artifacts" / "encoder-export"
    out_dir: Path = REPO_ROOT / "artifacts" / "tables"
    # Full set matches DESIGN §Putting-it-together: archetype
    # centroids want breadth, not recency.
    train_split: str = "train.evaluator.jsonl"
    k_archetypes: int = DEFAULT_K
    seed: int = 0


@dataclass(frozen=True, slots=True)
class BuildTablesResult:
    out_dir: Path
    play_frequency_path: Path
    archetype_centroids_path: Path
    manifest_path: Path
    n_decks: int
    k_effective: int


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf8"))
    return payload


def build_tables(opts: BuildTablesOptions | None = None) -> BuildTablesResult:
    opts = opts or BuildTablesOptions()
    prepared = opts.prepared_dir.resolve()
    encoder_export = opts.encoder_export_dir.resolve()

    prepare_manifest_path = prepared / "manifest.json"
    if not prepare_manifest_path.exists():
        raise FileNotFoundError(f"{prepare_manifest_path} not found.")
    encoder_manifest_path = encoder_export / "encoder-manifest.json"
    if not encoder_manifest_path.exists():
        raise FileNotFoundError(f"{encoder_manifest_path} not found.")
    prepare_manifest = _load_json(prepare_manifest_path)
    encoder_manifest = _load_json(encoder_manifest_path)

    vocab_size = load_vocab_size(prepared / "vocab.json")
    embeddings = load_card_embeddings(encoder_export / "card_embeddings.fp32.safetensors")
    if embeddings.shape[0] != vocab_size + 1:
        raise ValueError(
            f"encoder-export rows {embeddings.shape[0]} != vocab_size + 1 ({vocab_size + 1}).",
        )

    split_path = prepared / opts.train_split
    if not split_path.exists():
        raise FileNotFoundError(f"{split_path} not found.")
    decks = load_decks_jsonl(split_path)
    if not decks:
        raise RuntimeError(f"{opts.train_split} is empty.")

    opts.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Play-frequency table ---
    freq = compute_play_frequency(decks)
    freq_path = opts.out_dir / "play_frequency.json"
    write_play_frequency_json(freq, freq_path)

    # --- Archetype centroids ---
    deck_vectors = compute_deck_vectors(decks, embeddings)
    arch = kmeans(deck_vectors, k=opts.k_archetypes, seed=opts.seed)
    arch_path = opts.out_dir / "archetype_centroids.json"
    write_archetype_centroids_json(arch, arch_path)

    # --- Manifest ---
    manifest = {
        "generatedAt": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(opts).items()},
        "sources": {
            "prepared": str(prepared),
            "prepareContentHash": prepare_manifest.get("contentHash"),
            "cardsReleaseTag": prepare_manifest.get("sources", {}).get("cardsReleaseTag"),
            "cardSetVersion": prepare_manifest.get("sources", {}).get("cardSetVersion"),
            "encoderExport": str(encoder_export),
            "encoderManifest": encoder_manifest,
        },
        "vocabSize": vocab_size,
        "playFrequency": {
            "path": freq_path.name,
            "sha256": _sha256(freq_path),
            "inkPairCount": len(freq),
        },
        "archetypeCentroids": {
            "path": arch_path.name,
            "sha256": _sha256(arch_path),
            "k": int(arch.centroids.shape[0]),
            "dim": int(arch.centroids.shape[1]),
            "iterations": arch.iterations,
            "clusterSizes": list(arch.cluster_sizes),
        },
        "splits": {"trainDecks": len(decks)},
    }
    manifest_path = opts.out_dir / "tables-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")

    return BuildTablesResult(
        out_dir=opts.out_dir,
        play_frequency_path=freq_path,
        archetype_centroids_path=arch_path,
        manifest_path=manifest_path,
        n_decks=len(decks),
        k_effective=int(arch.centroids.shape[0]),
    )
