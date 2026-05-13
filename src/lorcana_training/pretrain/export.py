"""Export card embeddings from a trained pretrain checkpoint.

After ``lorcana-train pretrain-encoder`` finishes it leaves behind

    <out>/encoder.pt           torch state dict for the best epoch
    <out>/tokeniser.json       BPE tokeniser
    <out>/pretrain-manifest.json

``export_card_embeddings`` consumes those + the same ``cards-vN`` that
was pinned at training time and produces

    <export_out>/card_embeddings.fp32.safetensors     (N+1, 256) float32
    <export_out>/encoder_weights.safetensors          weights-only save
    <export_out>/tokeniser.json                       (copied for self-contained artifact)
    <export_out>/encoder-manifest.json

Row 0 in ``card_embeddings.fp32.safetensors`` is reserved for PAD and
stays all-zero (the embedder rows match the prepare vocab indexing).
Rows 1..N contain the L2-normalised card vectors for each logical
card, produced by a single forward pass in ``torch.no_grad``.

The ``encoder-manifest.json`` carries the full provenance chain —
prepare content-hash, pretrain checkpoint path + best-epoch loss,
tokeniser hash, cards tag + cardSetVersion — so downstream consumers
(proposal/evaluator training, lorcana-web) can reject a mismatched
combination deterministically.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file as save_numpy_file
from safetensors.torch import save_file as save_torch_file

from ..cards.download import download_cards
from ..cards.features import build_feature_schema, build_features
from ..cards.logical import build_logical_cards
from ..cards.vocab import build_vocab
from ..config import REPO_ROOT, load_config
from ..models.card_encoder import CardEncoder, CardEncoderConfig
from ..text import PAD_TOKEN, load_tokeniser, normalise_card_text


@dataclass(frozen=True, slots=True)
class ExportOptions:
    checkpoint_dir: Path = REPO_ROOT / "artifacts" / "encoder"
    out_dir: Path = REPO_ROOT / "artifacts" / "encoder-export"
    device: str | None = None  # None = auto (cuda > mps > cpu)
    batch_size: int = 64


@dataclass(frozen=True, slots=True)
class ExportResult:
    out_dir: Path
    embedding_shape: tuple[int, int]
    card_count: int
    manifest_path: Path


def _pick_device(pref: str | None) -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _load_checkpoint(
    checkpoint_dir: Path,
) -> tuple[CardEncoderConfig, dict[str, torch.Tensor]]:
    """Rebuild the encoder's config + weights from a checkpoint directory."""
    checkpoint_path = checkpoint_dir / "encoder.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"encoder.pt not found in {checkpoint_dir}")
    # weights_only=False needed because the checkpoint carries a dict of
    # torch.save'd state + our own python-level encoder_config dict.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt["encoder_config"]
    encoder_cfg = CardEncoderConfig(**cfg_dict)
    return encoder_cfg, ckpt["encoder"]


def _encode_all_cards(
    encoder: CardEncoder,
    *,
    texts: list[str],
    features: np.ndarray,
    tokeniser_path: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Run the encoder over every logical card; return (N+1, D) fp32.

    Row 0 is PAD (all zeros). Rows 1..N map 1:1 with ``texts`` / feature
    rows 1..N from prepare.
    """
    from ..text import load_tokeniser as _load_tok  # local alias for clarity

    tok = _load_tok(tokeniser_path)
    pad_id = tok.token_to_id(PAD_TOKEN)
    max_positions = encoder.cfg.max_positions
    n = len(texts)

    out = np.zeros((n + 1, encoder.cfg.encoder_dim), dtype=np.float32)
    encoder.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_texts = texts[start:end]
            token_ids = torch.full((end - start, max_positions), pad_id, dtype=torch.long)
            for i, t in enumerate(batch_texts):
                # Texts are already normalised at dataset build time;
                # apply again defensively so a caller invoking this
                # without prepare can't accidentally feed raw reminders.
                ids = tok.encode(normalise_card_text(t)).ids[:max_positions]
                token_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            struct_batch = torch.from_numpy(features[start + 1 : end + 1].copy())
            emb = encoder(token_ids.to(device), struct_batch.to(device))
            out[start + 1 : end + 1] = emb.detach().cpu().numpy()
    return out


def export_card_embeddings(
    opts: ExportOptions | None = None,
) -> ExportResult:
    opts = opts or ExportOptions()
    device = _pick_device(opts.device)

    # --- Load checkpoint + config. ---
    encoder_cfg, encoder_state = _load_checkpoint(opts.checkpoint_dir)

    # Sanity-check tokeniser + encoder vocab match *before* we allocate
    # any weights — a mismatch means someone retrained the tokeniser
    # after checkpointing the encoder, and silently loading it would
    # produce garbage embeddings.
    tokeniser_path = opts.checkpoint_dir / "tokeniser.json"
    if not tokeniser_path.exists():
        raise FileNotFoundError(f"tokeniser.json not found in {opts.checkpoint_dir}")
    tok = load_tokeniser(tokeniser_path)
    if tok.get_vocab_size() != encoder_cfg.vocab_size:
        raise ValueError(
            f"tokeniser vocab_size ({tok.get_vocab_size()}) does not match "
            f"encoder checkpoint vocab_size ({encoder_cfg.vocab_size})"
        )

    encoder = CardEncoder(encoder_cfg).to(device)
    encoder.load_state_dict(encoder_state)

    # --- Rebuild the pinned card pool + structured features. ---
    cfg = load_config()
    _, card_set = download_cards(cfg.scraper_repo, cfg.cards_release_tag)
    logical = build_logical_cards(card_set)
    vocab = build_vocab(logical)
    schema = build_feature_schema(logical.cards)
    features = build_features(vocab, schema)

    texts = [normalise_card_text(c.canonical.text or "") for c in logical.cards]

    embeddings = _encode_all_cards(
        encoder,
        texts=texts,
        features=features,
        tokeniser_path=tokeniser_path,
        device=device,
        batch_size=opts.batch_size,
    )

    # --- Write artifacts. ---
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = opts.out_dir / "card_embeddings.fp32.safetensors"
    save_numpy_file({"card_embeddings": embeddings}, str(embeddings_path))

    weights_path = opts.out_dir / "encoder_weights.safetensors"
    save_torch_file({k: v.detach().cpu() for k, v in encoder.state_dict().items()}, str(weights_path))

    out_tokeniser_path = opts.out_dir / "tokeniser.json"
    shutil.copyfile(tokeniser_path, out_tokeniser_path)

    # Read prepare + pretrain manifests for the provenance chain.
    pretrain_manifest_path = opts.checkpoint_dir / "pretrain-manifest.json"
    pretrain_manifest = (
        json.loads(pretrain_manifest_path.read_text(encoding="utf8"))
        if pretrain_manifest_path.exists()
        else {}
    )

    manifest = {
        "generatedAt": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "encoderConfig": {
            "vocabSize": encoder_cfg.vocab_size,
            "dModel": encoder_cfg.d_model,
            "nHeads": encoder_cfg.n_heads,
            "nLayers": encoder_cfg.n_layers,
            "maxPositions": encoder_cfg.max_positions,
            "textDim": encoder_cfg.text_dim,
            "structFeatureDim": encoder_cfg.struct_feature_dim,
            "structDim": encoder_cfg.struct_dim,
            "encoderDim": encoder_cfg.encoder_dim,
        },
        "embedding": {
            "rows": int(embeddings.shape[0]),
            "dim": int(embeddings.shape[1]),
            "dtype": "float32",
            "padRow": 0,
            "sha256": _sha256(embeddings_path),
        },
        "encoderWeights": {"sha256": _sha256(weights_path)},
        "tokeniser": {"sha256": _sha256(out_tokeniser_path)},
        "sources": {
            "cardsReleaseTag": cfg.cards_release_tag,
            "cardSetVersion": card_set.card_set_version,
            "cardsRepo": cfg.scraper_repo,
            "prepareContentHash": pretrain_manifest.get("sources", {}).get(
                "prepareContentHash"
            ),
            "pretrainBestEpoch": pretrain_manifest.get("bestEpoch"),
            "pretrainBestHeldoutTotal": pretrain_manifest.get("bestHeldoutTotal"),
        },
    }
    manifest_path = opts.out_dir / "encoder-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")

    return ExportResult(
        out_dir=opts.out_dir,
        embedding_shape=(int(embeddings.shape[0]), int(embeddings.shape[1])),
        card_count=len(logical.cards),
        manifest_path=manifest_path,
    )
