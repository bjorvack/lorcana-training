"""Training orchestration for ``lorcana-train train-proposal``.

Consumes:

  - ``<prepared>/train.proposal.jsonl`` — recency-filtered decks.
  - ``<prepared>/heldout.jsonl`` — held-out eval set (stratified by
    ink-pair × month; same file the evaluator and eval gauntlet read).
  - ``<prepared>/vocab.json`` — vocab size, ``cardSetVersion``.
  - ``<encoder_export>/card_embeddings.fp32.safetensors`` — frozen
    per-card R^256 vectors.

Produces:

  - ``<out>/proposal.pt``          best checkpoint (state_dict + config).
  - ``<out>/proposal-run.json``    per-epoch metrics + final selection.
  - ``<out>/proposal-manifest.json`` provenance (prepare hash,
                                     encoder-manifest hash, hyperparams).

Training objective:

    L = CE(softmax(logits), target)  −  β · H(softmax(logits))

Early-stops on held-out ``total_loss`` after ``patience`` epochs without
improvement, saving the best checkpoint by that metric. Device picked
automatically (cuda > mps > cpu) unless overridden.

The loop mirrors ``pretrain/run.py``'s structure so the two training
stages stay easy to read side-by-side: same cosine-warmup schedule,
same ``run.json`` history shape, same "save on improvement" gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..config import REPO_ROOT
from ..models.proposal import ProposalNet, ProposalNetConfig, proposal_loss
from .data import (
    ProposalDataset,
    collate_proposal,
    load_card_embeddings,
    load_decks_jsonl,
    load_vocab_size,
)


@dataclass(frozen=True, slots=True)
class ProposalOptions:
    # --- Paths ---
    prepared_dir: Path = REPO_ROOT / "prepared"
    encoder_export_dir: Path = REPO_ROOT / "artifacts" / "encoder-export"
    out_dir: Path = REPO_ROOT / "artifacts" / "proposal"
    # --- Training ---
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    patience: int = 5
    samples_per_deck: int = 12
    entropy_beta: float = 0.05
    # --- Model architecture (DESIGN defaults; rarely overridden) ---
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ff_dim: int = 1024
    dropout: float = 0.1
    freeze_card_embeddings: bool = True
    # --- Runtime ---
    device: str | None = None  # None = auto
    seed: int = 0
    num_workers: int = 0


@dataclass(frozen=True, slots=True)
class ProposalResult:
    out_dir: Path
    best_epoch: int
    best_heldout_total: float
    best_heldout_ce: float
    best_heldout_entropy: float
    gradient_parameter_count: int


def _pick_device(pref: str | None) -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cosine_with_warmup(optimiser: AdamW, total_steps: int, warmup_steps: int) -> LambdaLR:
    """Linear warmup → cosine decay, identical to pretrain/run.py.

    Kept as a local function rather than shared for the same reason
    the ``_masked_mean_max_pool`` helpers are duplicated: the two
    training stages are siblings, not a library, and independence
    makes each file readable on its own.
    """

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimiser, lr_lambda=lr_at)


@dataclass(frozen=True, slots=True)
class _LossAvg:
    total: float
    ce: float
    entropy: float


def _evaluate(
    *,
    model: ProposalNet,
    loader: "DataLoader[dict[str, torch.Tensor]]",
    device: torch.device,
    entropy_beta: float,
) -> _LossAvg:
    model.eval()
    total_total = 0.0
    total_ce = 0.0
    total_entropy = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            card_ids = batch["card_ids"].to(device)
            ink_multihot = batch["ink_multihot"].to(device)
            target_distribution = batch["target_distribution"].to(device)
            logits = model(card_ids, ink_multihot)
            total, ce, entropy = proposal_loss(
                logits,
                target_distribution,
                entropy_beta=entropy_beta,
            )
            total_total += total.item()
            total_ce += ce.item()
            total_entropy += entropy.item()
            n_batches += 1
    n = max(n_batches, 1)
    return _LossAvg(
        total=total_total / n,
        ce=total_ce / n,
        entropy=total_entropy / n,
    )


def _save_checkpoint(path: Path, *, model: ProposalNet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save only the trainable sub-modules. The card embedding buffer is
    # reconstructed from the encoder export at load time, so baking a
    # copy into every proposal checkpoint would be pure disk waste
    # (2 k × 256 × 4 B ≈ 2 MB per checkpoint) and also lets the same
    # proposal checkpoint be paired with an updated encoder export
    # (strict superset case).
    trainable_state = {
        k: v
        for k, v in model.state_dict().items()
        # ``card_embeddings`` shows up in the state dict even when
        # registered as a buffer. Skip it unconditionally.
        if not k.endswith("card_embeddings")
    }
    torch.save(
        {
            "trainable_state": trainable_state,
            "config": asdict(model.cfg),
        },
        path,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf8"))
    return payload


def _serialise_options(options_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert Path values to strings so the manifest stays JSON-safe."""
    out: dict[str, Any] = {}
    for k, v in options_dict.items():
        out[k] = str(v) if isinstance(v, Path) else v
    return out


def train_proposal(opts: ProposalOptions | None = None) -> ProposalResult:
    """Run the full proposal training loop; return best-checkpoint info."""
    opts = opts or ProposalOptions()
    torch.manual_seed(opts.seed)
    device = _pick_device(opts.device)

    # --- Resolve paths + provenance. -------------------------------
    prepared = opts.prepared_dir.resolve()
    prepare_manifest_path = prepared / "manifest.json"
    if not prepare_manifest_path.exists():
        raise FileNotFoundError(
            f"{prepare_manifest_path} not found. Run `lorcana-train prepare` first.",
        )
    prepare_manifest = _load_json(prepare_manifest_path)

    encoder_export = opts.encoder_export_dir.resolve()
    encoder_manifest_path = encoder_export / "encoder-manifest.json"
    if not encoder_manifest_path.exists():
        raise FileNotFoundError(
            f"{encoder_manifest_path} not found. Run `lorcana-train export-encoder` first.",
        )
    encoder_manifest = _load_json(encoder_manifest_path)

    # Cards-tag must match between prepare and encoder. A mismatch here
    # means the card embeddings were trained against a different vocab
    # than the deck dataset — training would still run but the ids
    # would silently point at wrong cards.
    prepare_cards_tag = prepare_manifest.get("sources", {}).get("cardsReleaseTag")
    encoder_cards_tag = encoder_manifest.get("sources", {}).get("cardsReleaseTag")
    if prepare_cards_tag and encoder_cards_tag and prepare_cards_tag != encoder_cards_tag:
        raise ValueError(
            "Cards-tag mismatch: "
            f"prepared={prepare_cards_tag!r}, encoder={encoder_cards_tag!r}. "
            "Re-run prepare or export-encoder against a consistent cards-vN.",
        )

    # --- Load vocabulary size + card embeddings. -------------------
    vocab_size = load_vocab_size(prepared / "vocab.json")
    card_embeddings_path = encoder_export / "card_embeddings.fp32.safetensors"
    card_embeddings = load_card_embeddings(card_embeddings_path)
    if card_embeddings.shape[0] != vocab_size + 1:
        raise ValueError(
            f"{card_embeddings_path}: {card_embeddings.shape[0]} rows "
            f"!= vocab_size + 1 ({vocab_size + 1}). Vocab/encoder drift.",
        )

    # --- Datasets. -------------------------------------------------
    train_decks = load_decks_jsonl(prepared / "train.proposal.jsonl")
    heldout_decks = load_decks_jsonl(prepared / "heldout.jsonl")
    if not train_decks:
        raise RuntimeError(
            "train.proposal.jsonl is empty — check the recency filter in prepare.",
        )
    train_ds = ProposalDataset(
        train_decks,
        vocab_size=vocab_size,
        samples_per_deck=opts.samples_per_deck,
        seed=opts.seed,
    )
    # Heldout uses one sample per deck to keep evaluation cheap and
    # deterministic; varying masks at eval time adds noise to the
    # best-checkpoint selection signal without improving it.
    heldout_ds: ProposalDataset | None
    if heldout_decks:
        heldout_ds = ProposalDataset(
            heldout_decks,
            vocab_size=vocab_size,
            samples_per_deck=1,
            seed=opts.seed ^ 0xDEAD,
        )
    else:
        heldout_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=opts.batch_size,
        shuffle=True,
        collate_fn=collate_proposal,
        drop_last=False,
        num_workers=opts.num_workers,
    )
    heldout_loader: "DataLoader[dict[str, torch.Tensor]] | None" = None
    if heldout_ds is not None:
        # DataLoader is generic on the *element* type of its dataset,
        # but ``collate_fn`` rewrites each batch into the dict shape
        # the training loop iterates over. mypy's stub doesn't model
        # that rewrite, so the outer handle is typed to match what we
        # actually see on the for-loop side.
        heldout_loader = DataLoader(
            heldout_ds,  # type: ignore[arg-type]
            batch_size=opts.batch_size,
            shuffle=False,
            collate_fn=collate_proposal,
            drop_last=False,
            num_workers=opts.num_workers,
        )

    # --- Model + optimiser. ----------------------------------------
    cfg = ProposalNetConfig(
        vocab_size=vocab_size,
        embed_dim=card_embeddings.shape[1],
        d_model=opts.d_model,
        n_heads=opts.n_heads,
        n_layers=opts.n_layers,
        ff_dim=opts.ff_dim,
        dropout=opts.dropout,
        freeze_card_embeddings=opts.freeze_card_embeddings,
    )
    model = ProposalNet(cfg, card_embeddings=card_embeddings).to(device)
    # Only parameters with ``requires_grad=True`` belong in the
    # optimiser; the ``freeze_card_embeddings`` buffer skips grad
    # automatically, but double-filtering makes the intent explicit
    # for the fine-tuning case too.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimiser = AdamW(
        trainable_params,
        lr=opts.learning_rate,
        weight_decay=opts.weight_decay,
    )
    total_steps = max(1, opts.epochs * len(train_loader))
    warmup_steps = int(total_steps * opts.warmup_ratio)
    scheduler = _cosine_with_warmup(optimiser, total_steps=total_steps, warmup_steps=warmup_steps)

    # --- Training loop. --------------------------------------------
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = opts.out_dir / "proposal.pt"
    history: list[dict[str, float]] = []
    best_total = math.inf
    best_epoch = -1
    best_ce = math.inf
    best_entropy = 0.0
    epochs_since_improvement = 0
    run_start = time.monotonic()

    for epoch in range(1, opts.epochs + 1):
        model.train()
        train_total = 0.0
        train_ce = 0.0
        train_entropy = 0.0
        n_batches = 0
        for batch in train_loader:
            card_ids = batch["card_ids"].to(device)
            ink_multihot = batch["ink_multihot"].to(device)
            target_distribution = batch["target_distribution"].to(device)
            logits = model(card_ids, ink_multihot)
            total, ce, entropy = proposal_loss(
                logits,
                target_distribution,
                entropy_beta=opts.entropy_beta,
            )
            optimiser.zero_grad(set_to_none=True)
            total.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimiser.step()
            scheduler.step()
            train_total += total.item()
            train_ce += ce.item()
            train_entropy += entropy.item()
            n_batches += 1

        n = max(n_batches, 1)
        entry: dict[str, float] = {
            "epoch": epoch,
            "train_total": train_total / n,
            "train_ce": train_ce / n,
            "train_entropy": train_entropy / n,
            "lr": optimiser.param_groups[0]["lr"],
            "elapsed_s": time.monotonic() - run_start,
        }

        # Held-out eval (skipped if no heldout set — tiny prepare runs
        # without the stratified split still train; they just can't
        # early-stop on the right signal).
        selection_metric: float
        if heldout_loader is not None:
            heldout = _evaluate(
                model=model,
                loader=heldout_loader,
                device=device,
                entropy_beta=opts.entropy_beta,
            )
            entry["heldout_total"] = heldout.total
            entry["heldout_ce"] = heldout.ce
            entry["heldout_entropy"] = heldout.entropy
            selection_metric = heldout.total
        else:
            selection_metric = train_total / n

        history.append(entry)
        (opts.out_dir / "proposal-run.json").write_text(
            json.dumps({"history": history}, indent=2) + "\n",
            encoding="utf8",
        )

        improved = selection_metric < best_total - 1e-4
        if improved:
            best_total = selection_metric
            best_epoch = epoch
            if heldout_loader is not None:
                best_ce = entry.get("heldout_ce", math.inf)
                best_entropy = entry.get("heldout_entropy", 0.0)
            else:
                best_ce = train_ce / n
                best_entropy = train_entropy / n
            epochs_since_improvement = 0
            _save_checkpoint(checkpoint_path, model=model)
        else:
            epochs_since_improvement += 1

        marker = "(best)" if improved else ""
        if heldout_loader is not None:
            print(
                f"[proposal] epoch {epoch:>3}/{opts.epochs}  "
                f"train={entry['train_total']:.4f}  "
                f"heldout={entry['heldout_total']:.4f}  "
                f"H={entry['heldout_entropy']:.2f}  "
                f"lr={entry['lr']:.2e}  {marker}"
            )
        else:
            print(
                f"[proposal] epoch {epoch:>3}/{opts.epochs}  "
                f"train={entry['train_total']:.4f}  "
                f"H={entry['train_entropy']:.2f}  "
                f"lr={entry['lr']:.2e}  {marker}"
            )

        if epochs_since_improvement >= opts.patience:
            print(
                f"[proposal] early stop after {epoch} epochs (no improvement for {opts.patience})"
            )
            break

    # Guard: if no epoch ever improved (shouldn't happen, but finite
    # losses + cold init could theoretically land best at epoch 1 with
    # no improvement), we still want a checkpoint on disk so downstream
    # stages don't break.
    if best_epoch == -1:
        _save_checkpoint(checkpoint_path, model=model)

    # --- Manifest. -------------------------------------------------
    manifest = {
        "generatedAt": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "config": asdict(cfg),
        "options": _serialise_options(asdict(opts)),
        "gradientParameterCount": model.gradient_parameter_count,
        "bestEpoch": best_epoch,
        "bestHeldoutTotal": best_total,
        "bestHeldoutCrossEntropy": best_ce,
        "bestHeldoutEntropy": best_entropy,
        "sources": {
            "prepared": str(prepared),
            "prepareContentHash": prepare_manifest.get("contentHash"),
            "cardsReleaseTag": prepare_cards_tag,
            "cardSetVersion": prepare_manifest.get("sources", {}).get("cardSetVersion"),
            "encoderExport": str(encoder_export),
            "encoderManifest": encoder_manifest,
            "cardEmbeddingsSha256": _sha256(card_embeddings_path),
        },
        "splits": {
            "trainDecks": len(train_decks),
            "heldoutDecks": len(heldout_decks),
            "samplesPerDeck": opts.samples_per_deck,
        },
    }
    (opts.out_dir / "proposal-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf8",
    )

    return ProposalResult(
        out_dir=opts.out_dir,
        best_epoch=best_epoch,
        best_heldout_total=best_total,
        best_heldout_ce=best_ce,
        best_heldout_entropy=best_entropy,
        gradient_parameter_count=model.gradient_parameter_count,
    )
