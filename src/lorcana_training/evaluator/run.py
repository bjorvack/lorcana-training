"""Training orchestration for ``lorcana-train train-evaluator``.

Consumes:

  - ``<prepared>/train.evaluator.jsonl`` — every validated deck.
    DESIGN.md §3 specifies the evaluator should train on the full
    set (no 12-month filter), so unlike the proposal stage there's
    no split to override.
  - ``<prepared>/heldout.jsonl`` — ink-pair × month-stratified eval
    slice. Same one the proposal net uses.
  - ``<prepared>/vocab.json`` / ``card_features.safetensors`` /
    ``feature_schema.json`` — for the :class:`CardIndex` the
    curriculum samplers need.
  - ``<encoder_export>/card_embeddings.fp32.safetensors`` — frozen
    R^256 card vectors, shared with the proposal net.

Produces:

  - ``<out>/evaluator.pt`` — best checkpoint (state_dict + config).
  - ``<out>/evaluator-run.json`` — per-epoch metrics.
  - ``<out>/evaluator-manifest.json`` — provenance chain.

Training uses a 3-phase curriculum (see :class:`CurriculumPhase`):

  - Epoch 1..N1:     RANDOM_IN_INK
  - Epoch N1+1..N2:  CURVE_MATCHED
  - Epoch N2+1..end: LOCAL_SWAP

The schedule is advisory — a caller that wants to skip a phase can
set the boundary at 0. ``warmup_epochs`` + ``curve_epochs`` +
``local_epochs`` is the total epoch count. Held-out is *always*
evaluated in CURVE_MATCHED mode because that's the hardest negative
the model sees in phase 2 and gives a stable comparator across
runs (local-swap negatives depend on the deck split).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..config import REPO_ROOT
from ..models.evaluator import Evaluator, EvaluatorConfig, evaluator_loss
from ..proposal.data import load_card_embeddings, load_decks_jsonl, load_vocab_size
from .data import (
    CurriculumPhase,
    EvaluatorDataset,
    build_card_index,
    collate_evaluator,
)


@dataclass(frozen=True, slots=True)
class EvaluatorOptions:
    # --- Paths ---
    prepared_dir: Path = REPO_ROOT / "prepared"
    encoder_export_dir: Path = REPO_ROOT / "artifacts" / "encoder-export"
    out_dir: Path = REPO_ROOT / "artifacts" / "evaluator"
    # --- Data ---
    # Always the wider split — the evaluator doesn't have the
    # meta-drift concern the proposal net originally did, and
    # DESIGN.md §3 prescribes "all in-vocab decks" explicitly.
    train_split: str = "train.evaluator.jsonl"
    samples_per_deck: int = 12
    # --- Curriculum schedule ---
    warmup_epochs: int = 5  # RANDOM_IN_INK
    curve_epochs: int = 5  # CURVE_MATCHED
    local_epochs: int = 5  # LOCAL_SWAP
    # --- Optimisation ---
    batch_size: int = 64  # 2× bigger than proposal: examples are cheaper
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    patience: int = 5
    # --- Model architecture (DESIGN defaults) ---
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 512
    dropout: float = 0.1
    freeze_card_embeddings: bool = True
    # --- Runtime ---
    device: str | None = None
    seed: int = 0
    num_workers: int = 0

    @property
    def total_epochs(self) -> int:
        return self.warmup_epochs + self.curve_epochs + self.local_epochs


@dataclass(frozen=True, slots=True)
class EvaluatorResult:
    out_dir: Path
    best_epoch: int
    best_heldout_bce: float
    best_heldout_auc: float
    best_phase: str
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
    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimiser, lr_lambda=lr_at)


def _phase_for_epoch(epoch: int, opts: EvaluatorOptions) -> CurriculumPhase:
    # ``epoch`` is 1-indexed by the training loop.
    if epoch <= opts.warmup_epochs:
        return CurriculumPhase.RANDOM_IN_INK
    if epoch <= opts.warmup_epochs + opts.curve_epochs:
        return CurriculumPhase.CURVE_MATCHED
    return CurriculumPhase.LOCAL_SWAP


@dataclass(frozen=True, slots=True)
class _EvalMetrics:
    bce: float
    auc: float
    accuracy: float


def _roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Mann–Whitney U form of ROC-AUC. Pure-tensor, no sklearn.

    Doesn't handle ties in the strict "0.5 per tie" sense, but a
    single-threshold discriminator rarely produces exact ties in
    enough pairs to matter for model selection. Returns NaN if the
    label set is degenerate (all positive or all negative).
    """
    labels = labels.to(dtype=torch.bool)
    pos = scores[labels]
    neg = scores[~labels]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    # (#positives with higher score than negative) / (|pos| × |neg|)
    greater = (pos.unsqueeze(1) > neg.unsqueeze(0)).sum()
    equal = (pos.unsqueeze(1) == neg.unsqueeze(0)).sum()
    return float((greater + 0.5 * equal) / (pos.numel() * neg.numel()))


def _evaluate(
    *,
    model: Evaluator,
    loader: "DataLoader[dict[str, torch.Tensor]]",
    device: torch.device,
) -> _EvalMetrics:
    model.eval()
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_bce = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            partial_ids = batch["partial_ids"].to(device)
            candidate_ids = batch["candidate_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(partial_ids, candidate_ids)
            total_bce += evaluator_loss(logits, labels).item()
            all_scores.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            n_batches += 1
    scores = torch.cat(all_scores) if all_scores else torch.empty(0)
    labels = torch.cat(all_labels) if all_labels else torch.empty(0)
    if scores.numel() == 0:
        return _EvalMetrics(bce=float("nan"), auc=float("nan"), accuracy=float("nan"))
    probs = torch.sigmoid(scores)
    predictions = (probs >= 0.5).float()
    accuracy = float((predictions == labels).float().mean())
    return _EvalMetrics(
        bce=total_bce / max(n_batches, 1),
        auc=_roc_auc(scores, labels),
        accuracy=accuracy,
    )


def _save_checkpoint(path: Path, *, model: Evaluator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_state = {
        k: v for k, v in model.state_dict().items() if not k.endswith("card_embeddings")
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
    out: dict[str, Any] = {}
    for k, v in options_dict.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, Enum):
            out[k] = v.value
        else:
            out[k] = v
    return out


def train_evaluator(opts: EvaluatorOptions | None = None) -> EvaluatorResult:
    """Run the full evaluator training loop; return best-checkpoint info."""
    opts = opts or EvaluatorOptions()
    torch.manual_seed(opts.seed)
    device = _pick_device(opts.device)

    # --- Provenance -----------------------------------------------
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

    prepare_cards_tag = prepare_manifest.get("sources", {}).get("cardsReleaseTag")
    encoder_cards_tag = encoder_manifest.get("sources", {}).get("cardsReleaseTag")
    if prepare_cards_tag and encoder_cards_tag and prepare_cards_tag != encoder_cards_tag:
        raise ValueError(
            f"Cards-tag mismatch: prepared={prepare_cards_tag!r}, encoder={encoder_cards_tag!r}.",
        )

    # --- Embeddings + vocab + card index --------------------------
    vocab_size = load_vocab_size(prepared / "vocab.json")
    card_embeddings_path = encoder_export / "card_embeddings.fp32.safetensors"
    card_embeddings = load_card_embeddings(card_embeddings_path)
    if card_embeddings.shape[0] != vocab_size + 1:
        raise ValueError(
            f"{card_embeddings_path}: {card_embeddings.shape[0]} rows "
            f"!= vocab_size + 1 ({vocab_size + 1}). Vocab/encoder drift.",
        )
    card_index = build_card_index(
        card_features_path=prepared / "card_features.safetensors",
        feature_schema_path=prepared / "feature_schema.json",
    )

    # --- Datasets -------------------------------------------------
    train_path = prepared / opts.train_split
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found.")
    train_decks = load_decks_jsonl(train_path)
    heldout_decks = load_decks_jsonl(prepared / "heldout.jsonl")
    if not train_decks:
        raise RuntimeError(f"{opts.train_split} is empty.")

    train_ds = EvaluatorDataset(
        train_decks,
        card_index=card_index,
        samples_per_deck=opts.samples_per_deck,
        initial_phase=CurriculumPhase.RANDOM_IN_INK,
        seed=opts.seed,
    )
    heldout_ds: EvaluatorDataset | None = None
    if heldout_decks:
        # Held-out always uses CURVE_MATCHED negatives so the
        # selection metric doesn't drift with the training phase.
        heldout_ds = EvaluatorDataset(
            heldout_decks,
            card_index=card_index,
            samples_per_deck=1,
            initial_phase=CurriculumPhase.CURVE_MATCHED,
            seed=opts.seed ^ 0xBEEF,
        )

    train_loader: "DataLoader[dict[str, torch.Tensor]]" = DataLoader(
        train_ds,  # type: ignore[arg-type]
        batch_size=opts.batch_size,
        shuffle=True,
        collate_fn=collate_evaluator,
        drop_last=False,
        num_workers=opts.num_workers,
    )
    heldout_loader: "DataLoader[dict[str, torch.Tensor]] | None" = None
    if heldout_ds is not None:
        heldout_loader = DataLoader(
            heldout_ds,  # type: ignore[arg-type]
            batch_size=opts.batch_size,
            shuffle=False,
            collate_fn=collate_evaluator,
            drop_last=False,
            num_workers=opts.num_workers,
        )

    # --- Model + optimiser ----------------------------------------
    cfg = EvaluatorConfig(
        vocab_size=vocab_size,
        embed_dim=card_embeddings.shape[1],
        d_model=opts.d_model,
        n_heads=opts.n_heads,
        n_layers=opts.n_layers,
        ff_dim=opts.ff_dim,
        dropout=opts.dropout,
        freeze_card_embeddings=opts.freeze_card_embeddings,
    )
    model = Evaluator(cfg, card_embeddings=card_embeddings).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimiser = AdamW(trainable_params, lr=opts.learning_rate, weight_decay=opts.weight_decay)
    total_epochs = opts.total_epochs
    total_steps = max(1, total_epochs * len(train_loader))
    warmup_steps = int(total_steps * opts.warmup_ratio)
    scheduler = _cosine_with_warmup(optimiser, total_steps=total_steps, warmup_steps=warmup_steps)

    # --- Training loop --------------------------------------------
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = opts.out_dir / "evaluator.pt"
    history: list[dict[str, Any]] = []
    best_bce = math.inf
    best_epoch = -1
    best_auc = 0.0
    best_phase = CurriculumPhase.RANDOM_IN_INK
    epochs_since_improvement = 0
    run_start = time.monotonic()

    for epoch in range(1, total_epochs + 1):
        phase = _phase_for_epoch(epoch, opts)
        train_ds.set_phase(phase)

        model.train()
        train_bce_total = 0.0
        n_batches = 0
        for batch in train_loader:
            partial_ids = batch["partial_ids"].to(device)
            candidate_ids = batch["candidate_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(partial_ids, candidate_ids)
            loss = evaluator_loss(logits, labels)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimiser.step()
            scheduler.step()
            train_bce_total += loss.item()
            n_batches += 1

        n = max(n_batches, 1)
        entry: dict[str, Any] = {
            "epoch": epoch,
            "phase": phase.value,
            "train_bce": train_bce_total / n,
            "lr": optimiser.param_groups[0]["lr"],
            "elapsed_s": time.monotonic() - run_start,
        }

        selection_metric: float
        if heldout_loader is not None:
            heldout = _evaluate(model=model, loader=heldout_loader, device=device)
            entry["heldout_bce"] = heldout.bce
            entry["heldout_auc"] = heldout.auc
            entry["heldout_accuracy"] = heldout.accuracy
            selection_metric = heldout.bce
        else:
            selection_metric = train_bce_total / n

        history.append(entry)
        (opts.out_dir / "evaluator-run.json").write_text(
            json.dumps({"history": history}, indent=2) + "\n", encoding="utf8"
        )

        improved = selection_metric < best_bce - 1e-4
        if improved:
            best_bce = selection_metric
            best_epoch = epoch
            best_phase = phase
            if heldout_loader is not None:
                best_auc = float(entry.get("heldout_auc", 0.0))
            epochs_since_improvement = 0
            _save_checkpoint(checkpoint_path, model=model)
        else:
            epochs_since_improvement += 1

        marker = "(best)" if improved else ""
        if heldout_loader is not None:
            print(
                f"[evaluator] epoch {epoch:>3}/{total_epochs}  phase={phase.value:<14}  "
                f"train_bce={entry['train_bce']:.4f}  "
                f"heldout_bce={entry['heldout_bce']:.4f}  "
                f"auc={entry['heldout_auc']:.3f}  "
                f"lr={entry['lr']:.2e}  {marker}"
            )
        else:
            print(
                f"[evaluator] epoch {epoch:>3}/{total_epochs}  phase={phase.value:<14}  "
                f"train_bce={entry['train_bce']:.4f}  "
                f"lr={entry['lr']:.2e}  {marker}"
            )

        if epochs_since_improvement >= opts.patience:
            print(
                f"[evaluator] early stop after {epoch} epochs (no improvement for {opts.patience})"
            )
            break

    if best_epoch == -1:
        _save_checkpoint(checkpoint_path, model=model)

    manifest = {
        "generatedAt": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "config": asdict(cfg),
        "options": _serialise_options(asdict(opts)),
        "gradientParameterCount": model.gradient_parameter_count,
        "bestEpoch": best_epoch,
        "bestHeldoutBce": best_bce,
        "bestHeldoutAuc": best_auc,
        "bestPhase": best_phase.value,
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
    (opts.out_dir / "evaluator-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8"
    )

    return EvaluatorResult(
        out_dir=opts.out_dir,
        best_epoch=best_epoch,
        best_heldout_bce=best_bce,
        best_heldout_auc=best_auc,
        best_phase=best_phase.value,
        gradient_parameter_count=model.gradient_parameter_count,
    )
