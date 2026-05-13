"""Top-level orchestration for the ``prepare`` pipeline stage.

Wires together the five commits that precede this one:

    cards-vN, tournaments-vN           (download)
            │
            ▼
    logical collapse + vocab           (cards.logical, cards.vocab)
            │
            ├── card_features.safetensors  (cards.features)
            │
            ▼
    validate decks (printing → logical, legality)   (dataset.validate)
            │
            ▼
    recency + stratified 90/10 splits  (dataset.splits)
            │
            ▼
    prepared/manifest.json             (this module)

The manifest records a content hash over (release tags, config knobs,
PREPARE_VERSION). Subsequent invocations with the same inputs short
-circuit after verifying the hash, so repeated CI runs are free. Pass
``force=True`` to override.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..cards.download import download_cards
from ..cards.features import build_feature_schema, build_features, write_features
from ..cards.logical import build_logical_cards
from ..cards.vocab import build_vocab, write_vocab
from ..config import REPO_ROOT, TrainingConfig, load_config
from ..dataset.download import download_tournaments
from ..dataset.splits import build_splits, write_splits
from ..dataset.validate import validate_dataset


# Bump this whenever the prepare output format changes in a way that
# older cached manifests should be invalidated. Otherwise a code bump
# could silently reuse stale artifacts.
PREPARE_VERSION = "1"


@dataclass(frozen=True, slots=True)
class PrepareOptions:
    out_dir: Path = REPO_ROOT / "prepared"
    strict_legality: bool = False
    proposal_recency_months: int = 12
    heldout_ratio: float = 0.10
    seed: int = 0
    force: bool = False


@dataclass(frozen=True, slots=True)
class PrepareResult:
    out_dir: Path
    content_hash: str
    cached: bool
    manifest_path: Path


def _compute_content_hash(cfg: TrainingConfig, opts: PrepareOptions) -> str:
    """Hash over every input that affects the prepared artifacts.

    Changing any of these should force a rebuild; unchanged combos
    deserve a cache hit.
    """
    payload = {
        "prepareVersion": PREPARE_VERSION,
        "schemasRepo": cfg.schemas_repo,
        "schemasReleaseTag": cfg.schemas_release_tag,
        "scraperRepo": cfg.scraper_repo,
        "cardsReleaseTag": cfg.cards_release_tag,
        "tournamentsReleaseTag": cfg.tournaments_release_tag,
        "strictLegality": opts.strict_legality,
        "proposalRecencyMonths": opts.proposal_recency_months,
        "heldoutRatio": opts.heldout_ratio,
        "seed": opts.seed,
    }
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(serialised.encode("utf8")).hexdigest()


def _load_cached_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_manifest(
    manifest_path: Path,
    *,
    content_hash: str,
    cfg: TrainingConfig,
    opts: PrepareOptions,
    card_set_version: str,
    validation_totals: dict[str, Any],
    split_totals: dict[str, Any],
    output_files: dict[str, Path],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "contentHash": content_hash,
        "prepareVersion": PREPARE_VERSION,
        "generatedAt": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "schemasRepo": cfg.schemas_repo,
            "schemasReleaseTag": cfg.schemas_release_tag,
            "scraperRepo": cfg.scraper_repo,
            "cardsReleaseTag": cfg.cards_release_tag,
            "cardSetVersion": card_set_version,
            "tournamentsReleaseTag": cfg.tournaments_release_tag,
        },
        "options": {
            "strictLegality": opts.strict_legality,
            "proposalRecencyMonths": opts.proposal_recency_months,
            "heldoutRatio": opts.heldout_ratio,
            "seed": opts.seed,
        },
        "validation": validation_totals,
        "splits": split_totals,
        "outputs": {
            name: str(path.relative_to(manifest_path.parent))
            for name, path in sorted(output_files.items())
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")


def prepare(
    opts: PrepareOptions | None = None,
    *,
    config: TrainingConfig | None = None,
) -> PrepareResult:
    """Run the full prepare stage.

    Returns a :class:`PrepareResult` with the manifest path and a
    ``cached`` flag indicating whether this invocation was a no-op.
    """
    opts = opts or PrepareOptions()
    cfg = config or load_config()
    out_dir = opts.out_dir.resolve()
    manifest_path = out_dir / "manifest.json"
    content_hash = _compute_content_hash(cfg, opts)

    cached = _load_cached_manifest(manifest_path)
    if cached is not None and cached.get("contentHash") == content_hash and not opts.force:
        return PrepareResult(
            out_dir=out_dir,
            content_hash=content_hash,
            cached=True,
            manifest_path=manifest_path,
        )

    # --- Download ---
    _, card_set = download_cards(cfg.scraper_repo, cfg.cards_release_tag)
    _, dataset = download_tournaments(cfg.scraper_repo, cfg.tournaments_release_tag)

    # --- Cards: logical collapse + vocab + structured features ---
    logical = build_logical_cards(card_set)
    vocab = build_vocab(logical)
    vocab_paths = write_vocab(
        vocab,
        logical,
        card_set,
        out_dir=out_dir,
        cards_release_tag=cfg.cards_release_tag,
    )

    schema = build_feature_schema(logical.cards)
    features = build_features(vocab, schema)
    feature_paths = write_features(features, schema, out_dir=out_dir)

    # --- Dataset: validate + split ---
    validated = validate_dataset(
        dataset,
        vocab=vocab,
        logical_cards=logical,
        strict_legality=opts.strict_legality,
    )
    splits = build_splits(
        validated,
        proposal_recency_months=opts.proposal_recency_months,
        heldout_ratio=opts.heldout_ratio,
        seed=opts.seed,
    )
    split_paths = write_splits(splits, out_dir=out_dir)

    # --- Manifest ---
    _write_manifest(
        manifest_path,
        content_hash=content_hash,
        cfg=cfg,
        opts=opts,
        card_set_version=card_set.card_set_version,
        validation_totals={
            "totalDecks": validated.report.total_decks,
            "validDecks": validated.report.valid_decks,
            "dropRate": validated.report.drop_rate,
            "dropReasons": dict(validated.report.drop_reasons_histogram),
        },
        split_totals={
            "trainProposal": splits.report.train_proposal,
            "trainEvaluator": splits.report.train_evaluator,
            "heldout": splits.report.heldout,
            "proposalCutoffDate": (
                splits.report.proposal_cutoff_date.isoformat()
                if splits.report.proposal_cutoff_date
                else None
            ),
            "strataCount": splits.report.strata_count,
        },
        output_files={**vocab_paths, **feature_paths, **split_paths},
    )

    return PrepareResult(
        out_dir=out_dir,
        content_hash=content_hash,
        cached=False,
        manifest_path=manifest_path,
    )
