"""Export trained proposal + evaluator checkpoints to ONNX.

Consumes:

  - ``<prepared>/vocab.json`` + ``card_features.safetensors`` (for
    vocab_size / cardSetVersion in the manifest).
  - ``<encoder_export>/card_embeddings.fp32.safetensors`` — the frozen
    R^256 table both checkpoints reference but don't save themselves.
  - ``<proposal>/proposal.pt`` + ``<evaluator>/evaluator.pt`` —
    trainable-sub-module state dicts saved by the two training stages.

Produces:

  - ``<out>/proposal.onnx``         Dynamic-axis ONNX graph.
  - ``<out>/evaluator.onnx``        Same.
  - ``<out>/card_embeddings.bin``   fp16 copy of the encoder's R^256
                                    table. Packed as a raw tensor
                                    (little-endian, (rows, 256) fp16)
                                    so the web app can mmap / slice by
                                    card id without parsing a format.
  - ``<out>/export-manifest.json``  Records input sha256s, ONNX opset,
                                    graph-level input / output names,
                                    and the downstream-expected input
                                    shape contracts.

Design notes:

- The card-embedding table is *not* baked into the ONNX graphs. We
  ship it as a sibling ``card_embeddings.bin`` for two reasons. First,
  onnxruntime-web loads the same table into both models once — baking
  it into each would double the download. Second, a future encoder
  re-export can swap the .bin without re-exporting the ONNX graphs,
  as long as the vocab size + embedding dim match.
- To make that work, we replace the models' ``card_embeddings``
  buffer with an input feed at export time. The web app reads the
  .bin, looks up the partial-deck card rows + the candidate row, and
  passes them in.
- Opset 17: covers scaled_dot_product_attention natively + is what
  recent onnxruntime-web ships. Older opsets drop attention to sub-
  ops which inflate graph size and run slower.
- Dynamic axes: batch + partial-deck length. Candidate is a scalar
  per-batch, so no dynamic axis there.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..config import REPO_ROOT
from ..models.evaluator import Evaluator, EvaluatorConfig
from ..models.proposal import ProposalNet, ProposalNetConfig
from ..proposal.data import load_card_embeddings, load_vocab_size


ONNX_OPSET = 17


@dataclass(frozen=True, slots=True)
class OnnxExportOptions:
    prepared_dir: Path = REPO_ROOT / "prepared"
    encoder_export_dir: Path = REPO_ROOT / "artifacts" / "encoder-export"
    proposal_dir: Path = REPO_ROOT / "artifacts" / "proposal"
    evaluator_dir: Path = REPO_ROOT / "artifacts" / "evaluator"
    out_dir: Path = REPO_ROOT / "artifacts" / "model-export"
    # Casting the embedding table to fp16 for the web bundle halves
    # the payload (a 2 283 × 256 fp16 is ~1.1 MB vs 2.2 MB fp32).
    # Torch inference + ONNX graphs still consume fp32 from the
    # card_embeddings feed — the web client up-casts on load.
    embeddings_dtype: str = "float16"


@dataclass(frozen=True, slots=True)
class OnnxExportResult:
    out_dir: Path
    proposal_path: Path
    evaluator_path: Path
    card_embeddings_path: Path
    manifest_path: Path


class _ProposalExport(nn.Module):
    """Proposal-net wrapper that takes card_embeddings as an input.

    The trained :class:`ProposalNet` holds the embedding table as a
    buffer, which ``torch.onnx.export`` would freeze into the graph
    as a Constant (bloating every model.onnx with a ~2 MB copy of
    something the web app already has). This wrapper rebuilds the
    same forward pass but reads the table from an input feed instead,
    so the exported graph carries only the Transformer + ink MLP +
    head weights.
    """

    def __init__(self, trained: ProposalNet) -> None:
        super().__init__()
        self.embed_projection = trained.embed_projection
        self.ink_embed = trained.ink_embed
        self.transformer = trained.transformer
        self.head = trained.head
        self.pad_token_id = trained.cfg.pad_token_id

    def forward(
        self,
        card_ids: torch.Tensor,
        ink_multihot: torch.Tensor,
        card_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        card_vecs = self.embed_projection(card_embeddings[card_ids])
        ink_vec = self.ink_embed(ink_multihot).unsqueeze(1)
        x = card_vecs + ink_vec
        padding_mask = card_ids == self.pad_token_id
        hidden = self.transformer(x, src_key_padding_mask=padding_mask)
        pooled = _masked_mean_max_pool(hidden, padding_mask)
        logits: torch.Tensor = self.head(pooled)
        return logits


class _EvaluatorExport(nn.Module):
    """Evaluator wrapper, same input-feed trick as :class:`_ProposalExport`."""

    def __init__(self, trained: Evaluator) -> None:
        super().__init__()
        self.embed_projection = trained.embed_projection
        self.deck_encoder = trained.deck_encoder
        self.candidate_mlp = trained.candidate_mlp
        self.head = trained.head
        self.pad_token_id = trained.cfg.pad_token_id

    def forward(
        self,
        partial_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        card_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        deck_vecs = self.embed_projection(card_embeddings[partial_ids])
        cand_vecs = self.embed_projection(card_embeddings[candidate_ids])
        padding_mask = partial_ids == self.pad_token_id
        deck_hidden = self.deck_encoder(deck_vecs, src_key_padding_mask=padding_mask)
        deck_pooled = _masked_mean_max_pool(deck_hidden, padding_mask)
        cand_vec = self.candidate_mlp(cand_vecs)
        fused = torch.cat([deck_pooled, cand_vec], dim=-1)
        logits: torch.Tensor = self.head(fused).squeeze(-1)
        return logits


def _masked_mean_max_pool(hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Duplicated from models/{proposal,evaluator} so the export
    wrapper doesn't reach into private module helpers."""
    mask = (~padding_mask).unsqueeze(-1).float()
    total = mask.sum(dim=1).clamp(min=1.0)
    mean = (hidden * mask).sum(dim=1) / total
    neg_inf = torch.finfo(hidden.dtype).min
    max_input = hidden.masked_fill(padding_mask.unsqueeze(-1), neg_inf)
    max_val, _ = max_input.max(dim=1)
    max_val = torch.where(padding_mask.all(dim=1, keepdim=True), torch.zeros_like(max_val), max_val)
    return torch.cat([mean, max_val], dim=-1)


def _load_proposal(checkpoint_path: Path, card_embeddings: torch.Tensor) -> ProposalNet:
    """Rebuild ProposalNet from ``proposal.pt`` + the encoder's
    embedding table. Uses the config serialised in the checkpoint so
    a future hyperparameter change is picked up automatically."""
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ProposalNetConfig(**blob["config"])
    model = ProposalNet(cfg, card_embeddings=card_embeddings)
    model.load_state_dict(blob["trainable_state"], strict=False)
    model.eval()
    return model


def _load_evaluator(checkpoint_path: Path, card_embeddings: torch.Tensor) -> Evaluator:
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = EvaluatorConfig(**blob["config"])
    model = Evaluator(cfg, card_embeddings=card_embeddings)
    model.load_state_dict(blob["trainable_state"], strict=False)
    model.eval()
    return model


def _export_proposal(
    model: ProposalNet,
    card_embeddings: torch.Tensor,
    out_path: Path,
) -> None:
    wrapper = _ProposalExport(model)
    wrapper.eval()
    vocab_plus_pad = card_embeddings.shape[0]
    # Dummy inputs at batch = 2 rather than batch = 1. Dynamo
    # sometimes eliminates axes of size 1 from the exported graph
    # even with dynamic_shapes set; passing batch = 2 keeps every
    # axis alive so the resulting ONNX genuinely accepts any batch
    # at runtime.
    dummy_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]], dtype=torch.long)
    dummy_ink = torch.zeros(2, 6, dtype=torch.float32)
    dummy_ink[0, 0] = 1.0
    dummy_ink[1, 1] = 1.0
    # The dynamo exporter with ``dynamic_shapes`` is the only
    # combination that handles torch.nn.TransformerEncoder's
    # internal reshapes at a batch / sequence length different from
    # the dummy input. The legacy exporter specialises those
    # reshapes to the traced numeric dims and produces a graph that
    # crashes at a different batch size.
    batch = torch.export.Dim("batch")
    partial_len = torch.export.Dim("partial_len", min=1)
    vocab = torch.export.Dim("vocab", min=2)
    dynamic_shapes_proposal = {
        "card_ids": {0: batch, 1: partial_len},
        "ink_multihot": {0: batch},
        "card_embeddings": {0: vocab},
    }
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_ink, card_embeddings),
        str(out_path),
        input_names=["card_ids", "ink_multihot", "card_embeddings"],
        output_names=["logits"],
        dynamic_shapes=dynamic_shapes_proposal,
        opset_version=ONNX_OPSET,
        dynamo=True,
    )
    del vocab_plus_pad  # suppress "unused" — kept for doc clarity


def _export_evaluator(
    model: Evaluator,
    card_embeddings: torch.Tensor,
    out_path: Path,
) -> None:
    wrapper = _EvaluatorExport(model)
    wrapper.eval()
    # Same batch = 2 trick as in _export_proposal.
    dummy_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]], dtype=torch.long)
    dummy_cand = torch.tensor([7, 9], dtype=torch.long)
    batch = torch.export.Dim("batch")
    partial_len = torch.export.Dim("partial_len", min=1)
    vocab = torch.export.Dim("vocab", min=2)
    dynamic_shapes_evaluator = {
        "partial_ids": {0: batch, 1: partial_len},
        "candidate_ids": {0: batch},
        "card_embeddings": {0: vocab},
    }
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_cand, card_embeddings),
        str(out_path),
        input_names=["partial_ids", "candidate_ids", "card_embeddings"],
        output_names=["logits"],
        dynamic_shapes=dynamic_shapes_evaluator,
        opset_version=ONNX_OPSET,
        dynamo=True,
    )


def _write_embeddings_bin(embeddings: torch.Tensor, out_path: Path, dtype: str) -> None:
    """Write the table as a contiguous raw tensor. Format is just:

        rows × cols × dtype little-endian bytes

    The web app does fetch('.bin') → ArrayBuffer → new Float16Array
    → reshape. No framing, no header — the shape lives in the
    manifest next to it.
    """
    if dtype not in {"float16", "float32"}:
        raise ValueError(f"embeddings_dtype must be float16 or float32, got {dtype}")
    cast_to = torch.float16 if dtype == "float16" else torch.float32
    out_path.write_bytes(embeddings.to(cast_to).contiguous().cpu().numpy().tobytes())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf8"))
    return payload


def export_models(opts: OnnxExportOptions | None = None) -> OnnxExportResult:
    """Export proposal + evaluator to ONNX plus a card-embeddings .bin."""
    opts = opts or OnnxExportOptions()

    # --- Load provenance from each upstream manifest. --------------
    prepared = opts.prepared_dir.resolve()
    encoder_export = opts.encoder_export_dir.resolve()
    proposal_dir = opts.proposal_dir.resolve()
    evaluator_dir = opts.evaluator_dir.resolve()

    for required in (
        prepared / "manifest.json",
        encoder_export / "encoder-manifest.json",
        proposal_dir / "proposal-manifest.json",
        proposal_dir / "proposal.pt",
        evaluator_dir / "evaluator-manifest.json",
        evaluator_dir / "evaluator.pt",
    ):
        if not required.exists():
            raise FileNotFoundError(f"{required} not found — did you run the upstream stage?")

    prepare_manifest = _load_json(prepared / "manifest.json")
    encoder_manifest = _load_json(encoder_export / "encoder-manifest.json")
    proposal_manifest = _load_json(proposal_dir / "proposal-manifest.json")
    evaluator_manifest = _load_json(evaluator_dir / "evaluator-manifest.json")

    # --- Embeddings shared between both models. --------------------
    vocab_size = load_vocab_size(prepared / "vocab.json")
    card_embeddings_path = encoder_export / "card_embeddings.fp32.safetensors"
    card_embeddings = load_card_embeddings(card_embeddings_path)
    if card_embeddings.shape[0] != vocab_size + 1:
        raise ValueError(
            f"{card_embeddings_path}: {card_embeddings.shape[0]} rows "
            f"!= vocab_size + 1 ({vocab_size + 1}).",
        )

    # --- Rebuild models, export to ONNX, write embedding .bin. -----
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    proposal_model = _load_proposal(proposal_dir / "proposal.pt", card_embeddings)
    evaluator_model = _load_evaluator(evaluator_dir / "evaluator.pt", card_embeddings)

    proposal_onnx = opts.out_dir / "proposal.onnx"
    evaluator_onnx = opts.out_dir / "evaluator.onnx"
    embeddings_bin = opts.out_dir / "card_embeddings.bin"

    _export_proposal(proposal_model, card_embeddings, proposal_onnx)
    _export_evaluator(evaluator_model, card_embeddings, evaluator_onnx)
    _write_embeddings_bin(card_embeddings, embeddings_bin, opts.embeddings_dtype)

    # --- Manifest ----------------------------------------------------
    # Dynamo-mode torch.onnx.export emits a sibling ``.onnx.data``
    # file for tensors over the 2 GB protobuf limit. Our models are
    # nowhere near that size but the dynamo exporter uses external
    # data unconditionally. Track the sidecar in the manifest so the
    # web bundle doesn't silently miss it.
    def _sidecar(p: Path) -> dict[str, Any] | None:
        sc = p.with_name(p.name + ".data")
        if sc.exists():
            return {"path": sc.name, "sha256": _sha256(sc), "bytes": sc.stat().st_size}
        return None

    manifest: dict[str, Any] = {
        "generatedAt": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "opset": ONNX_OPSET,
        "proposal": {
            "path": proposal_onnx.name,
            "sha256": _sha256(proposal_onnx),
            "externalData": _sidecar(proposal_onnx),
            "inputNames": ["card_ids", "ink_multihot", "card_embeddings"],
            "outputNames": ["logits"],
            "inputShapes": {
                "card_ids": ["batch", "partial_len"],
                "ink_multihot": ["batch", 6],
                "card_embeddings": ["vocab_plus_pad", card_embeddings.shape[1]],
            },
            "outputShapes": {
                "logits": ["batch", "vocab_plus_pad"],
            },
        },
        "evaluator": {
            "path": evaluator_onnx.name,
            "sha256": _sha256(evaluator_onnx),
            "externalData": _sidecar(evaluator_onnx),
            "inputNames": ["partial_ids", "candidate_ids", "card_embeddings"],
            "outputNames": ["logits"],
            "inputShapes": {
                "partial_ids": ["batch", "partial_len"],
                "candidate_ids": ["batch"],
                "card_embeddings": ["vocab_plus_pad", card_embeddings.shape[1]],
            },
            "outputShapes": {
                "logits": ["batch"],
            },
        },
        "cardEmbeddings": {
            "path": embeddings_bin.name,
            "sha256": _sha256(embeddings_bin),
            "rows": card_embeddings.shape[0],
            "dim": card_embeddings.shape[1],
            "dtype": opts.embeddings_dtype,
            "padRow": 0,
        },
        "vocabSize": vocab_size,
        "sources": {
            "prepared": str(prepared),
            "prepareContentHash": prepare_manifest.get("contentHash"),
            "cardsReleaseTag": prepare_manifest.get("sources", {}).get("cardsReleaseTag"),
            "cardSetVersion": prepare_manifest.get("sources", {}).get("cardSetVersion"),
            "encoderExport": str(encoder_export),
            "encoderManifest": encoder_manifest,
            "proposalManifest": proposal_manifest,
            "evaluatorManifest": evaluator_manifest,
        },
        "options": {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(opts).items()},
        },
    }
    manifest_path = opts.out_dir / "export-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")

    return OnnxExportResult(
        out_dir=opts.out_dir,
        proposal_path=proposal_onnx,
        evaluator_path=evaluator_onnx,
        card_embeddings_path=embeddings_bin,
        manifest_path=manifest_path,
    )
