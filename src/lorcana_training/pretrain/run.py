"""Training loop for ``lorcana-train pretrain-encoder``.

Two-head self-supervised pretrain (MLM on card text, denoising AE on
structured features). Optimises ``L_mlm + L_struct`` with AdamW and a
cosine LR schedule with warmup. Stops early when the held-out total
loss plateaus for ``patience`` epochs; saves the best checkpoint by
held-out total loss.

The loop is intentionally simple and device-agnostic:

- Picks CUDA if available, MPS on Apple Silicon, CPU otherwise. No
  distributed training; pretrain on ~2 k cards fits on a laptop.
- Mixed precision only on CUDA (MPS autocast works but isn't stable
  across torch versions, and CPU doesn't benefit).
- Per-epoch metrics written to ``run.json`` after every epoch so a
  crash leaves the partial history visible.

Output layout under ``<out_dir>/``:

    tokeniser.json         — BPE tokeniser (see text.tokeniser)
    encoder.pt             — best checkpoint's state_dict
    run.json               — per-epoch metrics + final selection
    pretrain-manifest.json — the sources pinned for this run (copy of
                             prepare's manifest plus the pretrain
                             hyperparams)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..cards.download import download_cards
from ..cards.features import FeatureSchema, build_feature_schema
from ..cards.logical import build_logical_cards
from ..cards.vocab import build_vocab
from ..config import REPO_ROOT, load_config
from ..models.card_encoder import CardEncoder, CardEncoderConfig
from ..models.pretrain_heads import (
    MlmHead,
    StructReconstructionHead,
    compute_pretrain_loss,
)
from ..text import SPECIAL_TOKENS
from ..train.masking import mask_structured_blocks, mask_tokens
from .data import (
    CardPretrainDataset,
    PreparedPaths,
    build_pretrain_dataset,
    build_pretrain_tokeniser,
    collate,
)


@dataclass(frozen=True, slots=True)
class PretrainOptions:
    prepared_dir: Path = REPO_ROOT / "prepared"
    out_dir: Path = REPO_ROOT / "artifacts" / "encoder"
    # Training hyperparameters.
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    struct_weight: float = 1.0
    # Encoder architecture knobs (override DESIGN defaults if desired).
    tokeniser_vocab_size: int = 32_000
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_dim: int = 512
    max_positions: int = 256
    text_dim: int = 192
    struct_hidden: int = 128
    struct_dim: int = 64
    encoder_dim: int = 256
    dropout: float = 0.1
    # Masking policy.
    token_mask_prob: float = 0.15
    struct_block_drop_prob: float = 0.3
    # Split + stopping.
    heldout_ratio: float = 0.10
    patience: int = 5
    # Device + seed.
    device: str | None = None  # None = auto
    seed: int = 0


@dataclass(frozen=True, slots=True)
class PretrainResult:
    out_dir: Path
    best_epoch: int
    best_heldout_total: float
    param_count: int


def _pick_device(pref: str | None) -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cosine_with_warmup(optimiser: AdamW, total_steps: int, warmup_steps: int) -> LambdaLR:
    """LR schedule: linear warmup to 1.0, then cosine decay to 0.0."""

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimiser, lr_lambda=lr_at)


def _build_encoder_config(opts: PretrainOptions, vocab_size: int, struct_dim: int) -> CardEncoderConfig:
    return CardEncoderConfig(
        vocab_size=vocab_size,
        pad_token_id=0,  # text.tokeniser guarantees [PAD] = 0
        d_model=opts.d_model,
        n_heads=opts.n_heads,
        n_layers=opts.n_layers,
        ff_dim=opts.ff_dim,
        max_positions=opts.max_positions,
        dropout=opts.dropout,
        text_dim=opts.text_dim,
        struct_feature_dim=struct_dim,
        struct_hidden=opts.struct_hidden,
        struct_dim=opts.struct_dim,
        encoder_dim=opts.encoder_dim,
    )


@dataclass(frozen=True, slots=True)
class _LossAvg:
    mlm: float
    struct: float
    total: float


def _evaluate(
    *,
    encoder: CardEncoder,
    mlm_head: MlmHead,
    struct_head: StructReconstructionHead,
    loader: DataLoader,
    schema: FeatureSchema,
    mask_token_id: int,
    special_ids: set[int],
    vocab_size: int,
    device: torch.device,
    struct_weight: float,
) -> _LossAvg:
    encoder.eval()
    mlm_head.eval()
    struct_head.eval()
    total_mlm = 0.0
    total_struct = 0.0
    total_total = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            token_ids = batch["token_ids"].to(device)
            struct_target = batch["struct_features"].to(device)
            # Masks use global torch RNG: device-specific torch.Generator
            # instances don't cross between CPU/MPS/CUDA, so we rely on
            # torch.manual_seed() in the caller for reproducibility.
            struct_input = mask_structured_blocks(
                struct_target, schema=schema, block_drop_prob=0.3
            ).features
            mlm = mask_tokens(
                token_ids,
                mask_token_id=mask_token_id,
                vocab_size=vocab_size,
                special_token_ids=special_ids,
            )
            losses = compute_pretrain_loss(
                encoder=encoder,
                mlm_head=mlm_head,
                struct_head=struct_head,
                token_ids=mlm.input_ids,
                mlm_labels=mlm.labels,
                struct_features_masked=struct_input,
                struct_features_target=struct_target,
                struct_weight=struct_weight,
            )
            total_mlm += losses.mlm.item()
            total_struct += losses.struct.item()
            total_total += losses.total.item()
            n_batches += 1
    n = max(n_batches, 1)
    return _LossAvg(mlm=total_mlm / n, struct=total_struct / n, total=total_total / n)


def _save_checkpoint(
    path: Path,
    *,
    encoder: CardEncoder,
    mlm_head: MlmHead,
    struct_head: StructReconstructionHead,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "mlm_head": mlm_head.state_dict(),
            "struct_head": struct_head.state_dict(),
            "encoder_config": asdict(encoder.cfg),
        },
        path,
    )


def pretrain_encoder(opts: PretrainOptions | None = None) -> PretrainResult:
    """Run the full pretrain loop; return best-checkpoint info."""
    opts = opts or PretrainOptions()
    torch.manual_seed(opts.seed)
    device = _pick_device(opts.device)

    # --- Inputs: rebuild vocab/features/cards from the prepared snapshot. ---
    prepared = opts.prepared_dir.resolve()
    if not (prepared / "manifest.json").exists():
        raise FileNotFoundError(
            f"{prepared}/manifest.json not found. Run `lorcana-train prepare` first."
        )
    prep_manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf8"))
    cfg = load_config()
    _, card_set = download_cards(cfg.scraper_repo, cfg.cards_release_tag)
    logical = build_logical_cards(card_set)
    vocab = build_vocab(logical)
    schema = build_feature_schema(logical.cards)

    # --- Tokeniser: trained in-process, then saved as part of output. ---
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    tokeniser = build_pretrain_tokeniser(
        logical,
        out_path=opts.out_dir / "tokeniser.json",
        vocab_size=opts.tokeniser_vocab_size,
    )
    vocab_size = tokeniser.get_vocab_size()
    mask_token_id = tokeniser.token_to_id("[MASK]")
    special_ids = {tokeniser.token_to_id(t) for t in SPECIAL_TOKENS}
    special_ids.discard(None)  # type: ignore[arg-type]

    # --- Dataset split. ---
    data = build_pretrain_dataset(
        PreparedPaths(
            vocab=prepared / "vocab.json",
            card_features=prepared / "card_features.safetensors",
            feature_schema=prepared / "feature_schema.json",
        ),
        logical_cards=logical,
        vocab=vocab,
        schema=schema,
        tokeniser=tokeniser,
        heldout_ratio=opts.heldout_ratio,
    )
    train_ds = CardPretrainDataset(data, indices=data.train_indices, max_positions=opts.max_positions)
    heldout_ds = CardPretrainDataset(
        data, indices=data.heldout_indices, max_positions=opts.max_positions
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=opts.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=False,
    )
    heldout_loader = DataLoader(
        heldout_ds,
        batch_size=opts.batch_size,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
    )

    # --- Model + heads. ---
    encoder_cfg = _build_encoder_config(opts, vocab_size=vocab_size, struct_dim=schema.dim)
    encoder = CardEncoder(encoder_cfg).to(device)
    mlm_head = MlmHead(encoder).to(device)
    struct_head = StructReconstructionHead(encoder_cfg).to(device)
    params = list(encoder.parameters()) + list(mlm_head.parameters()) + list(struct_head.parameters())
    optimiser = AdamW(params, lr=opts.learning_rate, weight_decay=opts.weight_decay)
    total_steps = max(1, opts.epochs * len(train_loader))
    warmup_steps = int(total_steps * opts.warmup_ratio)
    scheduler = _cosine_with_warmup(optimiser, total_steps=total_steps, warmup_steps=warmup_steps)

    # --- Training loop. ---
    # Masks use global torch RNG (see note in _evaluate).
    history: list[dict[str, float]] = []
    best_heldout = math.inf
    best_epoch = -1
    epochs_since_improvement = 0
    checkpoint_path = opts.out_dir / "encoder.pt"
    run_start = time.monotonic()

    for epoch in range(1, opts.epochs + 1):
        encoder.train()
        mlm_head.train()
        struct_head.train()
        train_total = 0.0
        train_mlm = 0.0
        train_struct = 0.0
        n_batches = 0
        for batch in train_loader:
            token_ids = batch["token_ids"].to(device)
            struct_target = batch["struct_features"].to(device)
            struct_input = mask_structured_blocks(
                struct_target,
                schema=schema,
                block_drop_prob=opts.struct_block_drop_prob,
            ).features
            mlm = mask_tokens(
                token_ids,
                mask_token_id=mask_token_id,
                vocab_size=vocab_size,
                special_token_ids=special_ids,
                mask_prob=opts.token_mask_prob,
            )
            losses = compute_pretrain_loss(
                encoder=encoder,
                mlm_head=mlm_head,
                struct_head=struct_head,
                token_ids=mlm.input_ids,
                mlm_labels=mlm.labels,
                struct_features_masked=struct_input,
                struct_features_target=struct_target,
                struct_weight=opts.struct_weight,
            )
            optimiser.zero_grad(set_to_none=True)
            losses.total.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimiser.step()
            scheduler.step()
            train_total += losses.total.item()
            train_mlm += losses.mlm.item()
            train_struct += losses.struct.item()
            n_batches += 1

        heldout = _evaluate(
            encoder=encoder,
            mlm_head=mlm_head,
            struct_head=struct_head,
            loader=heldout_loader,
            schema=schema,
            mask_token_id=mask_token_id,
            special_ids=special_ids,
            vocab_size=vocab_size,
            device=device,
            struct_weight=opts.struct_weight,
        )

        n = max(n_batches, 1)
        entry = {
            "epoch": epoch,
            "train_total": train_total / n,
            "train_mlm": train_mlm / n,
            "train_struct": train_struct / n,
            "heldout_total": heldout.total,
            "heldout_mlm": heldout.mlm,
            "heldout_struct": heldout.struct,
            "lr": optimiser.param_groups[0]["lr"],
            "elapsed_s": time.monotonic() - run_start,
        }
        history.append(entry)
        (opts.out_dir / "run.json").write_text(
            json.dumps({"history": history}, indent=2) + "\n", encoding="utf8"
        )

        improved = heldout.total < best_heldout - 1e-4
        if improved:
            best_heldout = heldout.total
            best_epoch = epoch
            epochs_since_improvement = 0
            _save_checkpoint(
                checkpoint_path,
                encoder=encoder,
                mlm_head=mlm_head,
                struct_head=struct_head,
            )
        else:
            epochs_since_improvement += 1

        print(
            f"[pretrain] epoch {epoch:>3}/{opts.epochs}  "
            f"train={entry['train_total']:.4f}  "
            f"heldout={entry['heldout_total']:.4f}  "
            f"lr={entry['lr']:.2e}  "
            f"{'(best)' if improved else ''}"
        )

        if epochs_since_improvement >= opts.patience:
            print(f"[pretrain] early stop after {epoch} epochs (no improvement for {opts.patience})")
            break

    # --- Manifest. ---
    param_count = sum(p.numel() for p in encoder.parameters())
    manifest = {
        "generatedAt": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "encoderConfig": asdict(encoder_cfg),
        "options": asdict(opts),
        "parameterCount": param_count,
        "bestEpoch": best_epoch,
        "bestHeldoutTotal": best_heldout,
        "tokeniserVocabSize": vocab_size,
        "sources": {
            "prepared": str(prepared),
            "prepareContentHash": prep_manifest.get("contentHash"),
            "cardsReleaseTag": prep_manifest.get("sources", {}).get("cardsReleaseTag"),
            "cardSetVersion": prep_manifest.get("sources", {}).get("cardSetVersion"),
        },
    }
    _serialise_options(manifest)
    (opts.out_dir / "pretrain-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8"
    )
    # Ensure the final checkpoint exists even if no epoch improved
    # (shouldn't happen with finite losses; guard the CLI anyway).
    if best_epoch == -1:
        _save_checkpoint(
            checkpoint_path,
            encoder=encoder,
            mlm_head=mlm_head,
            struct_head=struct_head,
        )

    return PretrainResult(
        out_dir=opts.out_dir,
        best_epoch=best_epoch,
        best_heldout_total=best_heldout,
        param_count=param_count,
    )


def _serialise_options(manifest: dict) -> None:
    """Convert Path values to strings so manifest.json is JSON-safe."""
    opts = manifest.get("options", {})
    for key, value in list(opts.items()):
        if isinstance(value, Path):
            opts[key] = str(value)
