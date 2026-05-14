"""End-to-end ONNX export smoke test.

Builds a tiny synthetic ``prepared/`` + ``encoder-export/`` +
proposal + evaluator checkpoint on disk, runs :func:`export_models`,
then verifies:

  - ONNX files exist + load via onnx.load.
  - onnxruntime can run both graphs and produce the declared output
    shapes against dynamic axes.
  - card_embeddings.bin has the expected byte size for the declared
    dtype.
  - The manifest records consistent shapes + sha256s.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from safetensors.numpy import save_file as save_numpy_file
from safetensors.torch import save_file as save_torch_file

from lorcana_training.export import OnnxExportOptions, export_models
from lorcana_training.models.evaluator import Evaluator, EvaluatorConfig
from lorcana_training.models.proposal import ProposalNet, ProposalNetConfig


def _build_inputs(tmp_path: Path, vocab_size: int, embed_dim: int) -> dict[str, Path]:
    prepared = tmp_path / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "vocab.json").write_text(
        json.dumps(
            {
                "padIndex": 0,
                "size": vocab_size,
                "cardSetVersion": "sha256:test",
                "cardsReleaseTag": "cards-vTEST",
            }
        ),
        encoding="utf8",
    )
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "contentHash": "sha256:abc",
                "sources": {
                    "cardsReleaseTag": "cards-vTEST",
                    "cardSetVersion": "sha256:test",
                },
            }
        ),
        encoding="utf8",
    )
    # The export step now reads card_features.safetensors +
    # feature_schema.json to derive a per-card ink mask. The test
    # fixture writes a minimal pair: vocab_size cards, each with a
    # single Amber ink bit so the mask payload is well-shaped.
    feature_schema = {
        "dim": 18,
        "slices": {"cost": [0, 12], "inks": [12, 6], "types": [0, 0]},
        "scalars": {},
        "classes": {
            "inks": ["Amber", "Amethyst", "Emerald", "Ruby", "Sapphire", "Steel"],
        },
        "normalisers": {},
    }
    (prepared / "feature_schema.json").write_text(json.dumps(feature_schema), encoding="utf8")
    features = torch.zeros(vocab_size + 1, feature_schema["dim"], dtype=torch.float32)
    # All cards "Amber" — enough for the ink-mask code path to run
    # without contaminating any other test invariant.
    features[1:, feature_schema["slices"]["inks"][0]] = 1.0
    from safetensors.torch import save_file as _save_st_torch

    _save_st_torch({"card_features": features}, str(prepared / "card_features.safetensors"))

    encoder = tmp_path / "encoder-export"
    encoder.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(vocab_size + 1, embed_dim, generator=g)
    save_numpy_file(
        {"card_embeddings": emb.numpy().astype("float32")},
        str(encoder / "card_embeddings.fp32.safetensors"),
    )
    (encoder / "encoder-manifest.json").write_text(
        json.dumps(
            {
                "sources": {
                    "cardsReleaseTag": "cards-vTEST",
                    "cardSetVersion": "sha256:test",
                },
                "embedding": {
                    "rows": vocab_size + 1,
                    "dim": embed_dim,
                    "dtype": "float32",
                },
            }
        ),
        encoding="utf8",
    )

    # Proposal checkpoint.
    proposal_cfg = ProposalNetConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        d_model=embed_dim,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        dropout=0.0,
    )
    proposal_model = ProposalNet(proposal_cfg, card_embeddings=emb)
    proposal_dir = tmp_path / "proposal"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    trainable = {
        k: v for k, v in proposal_model.state_dict().items() if not k.endswith("card_embeddings")
    }
    from dataclasses import asdict as _asdict

    torch.save(
        {"trainable_state": trainable, "config": _asdict(proposal_cfg)},
        proposal_dir / "proposal.pt",
    )
    (proposal_dir / "proposal-manifest.json").write_text(
        json.dumps({"bestEpoch": 1}),
        encoding="utf8",
    )
    # Keep ruff happy about unused import.
    _ = save_torch_file

    # Evaluator checkpoint.
    evaluator_cfg = EvaluatorConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        d_model=embed_dim,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        dropout=0.0,
    )
    evaluator_model = Evaluator(evaluator_cfg, card_embeddings=emb)
    evaluator_dir = tmp_path / "evaluator"
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    eval_trainable = {
        k: v for k, v in evaluator_model.state_dict().items() if not k.endswith("card_embeddings")
    }
    torch.save(
        {"trainable_state": eval_trainable, "config": _asdict(evaluator_cfg)},
        evaluator_dir / "evaluator.pt",
    )
    (evaluator_dir / "evaluator-manifest.json").write_text(
        json.dumps({"bestEpoch": 1}),
        encoding="utf8",
    )

    return {
        "prepared": prepared,
        "encoder": encoder,
        "proposal": proposal_dir,
        "evaluator": evaluator_dir,
    }


def test_onnx_export_end_to_end(tmp_path: Path) -> None:
    vocab_size = 12
    embed_dim = 16
    dirs = _build_inputs(tmp_path, vocab_size=vocab_size, embed_dim=embed_dim)
    opts = OnnxExportOptions(
        prepared_dir=dirs["prepared"],
        encoder_export_dir=dirs["encoder"],
        proposal_dir=dirs["proposal"],
        evaluator_dir=dirs["evaluator"],
        out_dir=tmp_path / "model-export",
        embeddings_dtype="float16",
    )
    result = export_models(opts)

    # All four outputs exist.
    assert result.proposal_path.exists()
    assert result.evaluator_path.exists()
    assert result.card_embeddings_path.exists()
    assert result.manifest_path.exists()

    # card_embeddings.bin size = (vocab_size + 1) × embed_dim × 2 bytes (fp16).
    expected = (vocab_size + 1) * embed_dim * 2
    assert result.card_embeddings_path.stat().st_size == expected

    # ONNX files are valid.
    onnx.checker.check_model(onnx.load(str(result.proposal_path)))
    onnx.checker.check_model(onnx.load(str(result.evaluator_path)))

    # ORT inference at a batch ≠ the dummy input's batch → verifies
    # dynamic axes actually work (not specialised to batch=1).
    embeddings = np.frombuffer(result.card_embeddings_path.read_bytes(), dtype=np.float16)
    embeddings = embeddings.reshape(vocab_size + 1, embed_dim).astype(np.float32)

    # Proposal.
    proposal_sess = ort.InferenceSession(str(result.proposal_path))
    card_ids = np.array([[1, 2, 3, 0], [4, 5, 0, 0], [6, 0, 0, 0]], dtype=np.int64)
    ink = np.zeros((3, 6), dtype=np.float32)
    ink[:, 0] = 1.0
    (prop_logits,) = proposal_sess.run(
        None,
        {
            "card_ids": card_ids,
            "ink_multihot": ink,
            "card_embeddings": embeddings,
        },
    )
    assert prop_logits.shape == (3, vocab_size + 1)
    assert np.isfinite(prop_logits).all()

    # Evaluator.
    eval_sess = ort.InferenceSession(str(result.evaluator_path))
    partial = np.array([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=np.int64)
    candidate = np.array([7, 9], dtype=np.int64)
    (eval_logits,) = eval_sess.run(
        None,
        {
            "partial_ids": partial,
            "candidate_ids": candidate,
            "card_embeddings": embeddings,
        },
    )
    assert eval_logits.shape == (2,)
    assert np.isfinite(eval_logits).all()

    # Manifest sanity.
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["opset"] == 17
    assert manifest["vocabSize"] == vocab_size
    assert manifest["cardEmbeddings"]["dtype"] == "float16"
    assert manifest["cardEmbeddings"]["rows"] == vocab_size + 1
    assert manifest["cardEmbeddings"]["dim"] == embed_dim
    # sha256 recorded and real.
    assert manifest["proposal"]["sha256"].startswith("sha256:")
    assert manifest["evaluator"]["sha256"].startswith("sha256:")


def test_onnx_export_numerical_equivalence(tmp_path: Path) -> None:
    """Torch-forward vs. ONNX-runtime should agree to within 1e-4 on a
    fresh deterministic input. A large divergence would mean something
    in the export wrapper's forward differs from the trained model's
    forward — the bug this test exists to catch."""
    torch.manual_seed(0)
    vocab_size = 10
    embed_dim = 16
    dirs = _build_inputs(tmp_path, vocab_size=vocab_size, embed_dim=embed_dim)
    opts = OnnxExportOptions(
        prepared_dir=dirs["prepared"],
        encoder_export_dir=dirs["encoder"],
        proposal_dir=dirs["proposal"],
        evaluator_dir=dirs["evaluator"],
        out_dir=tmp_path / "model-export",
        # fp32 to keep numerical drift below 1e-4 — fp16 embeddings
        # would round trip to ~1e-3 and we'd have to loosen the
        # tolerance beyond what's a real correctness signal.
        embeddings_dtype="float32",
    )
    result = export_models(opts)

    # Torch side: rebuild the same wrappers used at export time.
    from lorcana_training.export.to_onnx import (
        _EvaluatorExport,
        _ProposalExport,
        _load_evaluator,
        _load_proposal,
    )
    from lorcana_training.proposal.data import load_card_embeddings

    embeddings = load_card_embeddings(dirs["encoder"] / "card_embeddings.fp32.safetensors")
    torch_proposal = _ProposalExport(_load_proposal(dirs["proposal"] / "proposal.pt", embeddings))
    torch_evaluator = _EvaluatorExport(
        _load_evaluator(dirs["evaluator"] / "evaluator.pt", embeddings)
    )
    torch_proposal.eval()
    torch_evaluator.eval()

    card_ids = torch.tensor([[1, 2, 3, 0, 0]], dtype=torch.long)
    ink = torch.zeros(1, 6, dtype=torch.float32)
    ink[0, 0] = 1.0
    with torch.no_grad():
        torch_prop_logits = torch_proposal(card_ids, ink, embeddings).numpy()
        torch_eval_logits = torch_evaluator(
            card_ids, torch.tensor([5], dtype=torch.long), embeddings
        ).numpy()

    embeddings_np = embeddings.numpy()
    prop_sess = ort.InferenceSession(str(result.proposal_path))
    (ort_prop_logits,) = prop_sess.run(
        None,
        {
            "card_ids": card_ids.numpy(),
            "ink_multihot": ink.numpy(),
            "card_embeddings": embeddings_np,
        },
    )
    eval_sess = ort.InferenceSession(str(result.evaluator_path))
    (ort_eval_logits,) = eval_sess.run(
        None,
        {
            "partial_ids": card_ids.numpy(),
            "candidate_ids": np.array([5], dtype=np.int64),
            "card_embeddings": embeddings_np,
        },
    )

    np.testing.assert_allclose(ort_prop_logits, torch_prop_logits, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(ort_eval_logits, torch_eval_logits, atol=1e-4, rtol=1e-4)
