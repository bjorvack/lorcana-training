"""Tests for export_card_embeddings.

The full live path (train a real checkpoint + export) is gated on
``RUN_NETWORK_TESTS=1``; the fast suite stubs the network calls and
builds a tiny throwaway checkpoint so we can exercise the export
pipeline end-to-end without hitting GitHub.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from safetensors.numpy import load_file as load_numpy_file

from lorcana_training.cards.features import build_feature_schema
from lorcana_training.cards.logical import build_logical_cards
from lorcana_training.cards.vocab import build_vocab
from lorcana_training.models.card_encoder import CardEncoder, CardEncoderConfig
from lorcana_training.pretrain.export import (
    ExportOptions,
    export_card_embeddings,
)
from lorcana_training.schemas.generated.card_set import CardSet
from lorcana_training.text import collect_reserved_tokens, train_tokeniser


def _card(id_: str, name: str, text: str = "") -> dict:
    return {
        "id": id_,
        "name": name,
        "version": "Test",
        "setCode": "1",
        "cardNumber": int(id_.split("_")[-1]),
        "cost": 3,
        "inkwell": True,
        "inks": ["Amber"],
        "types": ["Character"],
        "classifications": ["Storyborn"],
        "keywords": [],
        "text": text,
        "flavor": None,
        "imageUrl": f"https://example.test/{id_}.avif",
        "legality": "legal",
        "lore": 2,
        "strength": 3,
        "willpower": 3,
        "moveCost": None,
    }


def _fake_card_set() -> CardSet:
    cards = [
        _card(f"crd_{i:02d}", f"Char{i}", text=f"Rush. When played gain {i} lore.")
        for i in range(8)
    ]
    return CardSet.model_validate(
        {"cardSetVersion": "sha256:fake", "fetchedAt": "2026-05-13T00:00:00Z", "cards": cards}
    )


def _small_cfg(vocab_size: int, struct_dim: int) -> CardEncoderConfig:
    return CardEncoderConfig(
        vocab_size=vocab_size,
        d_model=16,
        n_heads=4,
        n_layers=2,
        ff_dim=32,
        max_positions=32,
        text_dim=24,
        struct_feature_dim=struct_dim,
        struct_hidden=32,
        struct_dim=12,
        encoder_dim=48,
    )


def _build_fake_checkpoint(tmp_path: Path, cs: CardSet) -> tuple[Path, CardEncoderConfig]:
    """Spin up a random-weights encoder + tokeniser in tmp_path."""
    logical = build_logical_cards(cs)
    schema = build_feature_schema(logical.cards)
    build_vocab(logical)  # sanity side-effect — triggers PAD-row check
    tokeniser_path = tmp_path / "tokeniser.json"
    tok = train_tokeniser(
        [c.canonical.text for c in logical.cards if c.canonical.text],
        out_path=tokeniser_path,
        vocab_size=200,
        reserved_tokens=collect_reserved_tokens(logical.cards),
    )
    cfg = _small_cfg(vocab_size=tok.get_vocab_size(), struct_dim=schema.dim)
    encoder = CardEncoder(cfg)
    # No training: we just want the export to run over random weights.
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "mlm_head": {},
            "struct_head": {},
            "encoder_config": {
                "vocab_size": cfg.vocab_size,
                "pad_token_id": cfg.pad_token_id,
                "d_model": cfg.d_model,
                "n_heads": cfg.n_heads,
                "n_layers": cfg.n_layers,
                "ff_dim": cfg.ff_dim,
                "max_positions": cfg.max_positions,
                "dropout": cfg.dropout,
                "text_dim": cfg.text_dim,
                "struct_feature_dim": cfg.struct_feature_dim,
                "struct_hidden": cfg.struct_hidden,
                "struct_dim": cfg.struct_dim,
                "encoder_dim": cfg.encoder_dim,
            },
        },
        tmp_path / "encoder.pt",
    )
    (tmp_path / "pretrain-manifest.json").write_text(
        json.dumps(
            {
                "bestEpoch": 1,
                "bestHeldoutTotal": 0.5,
                "sources": {"prepareContentHash": "sha256:fake"},
            }
        ),
        encoding="utf8",
    )
    return tmp_path, cfg


def test_export_produces_expected_artifacts(tmp_path: Path) -> None:
    cs = _fake_card_set()
    checkpoint_dir, cfg = _build_fake_checkpoint(tmp_path / "ckpt", cs)
    out_dir = tmp_path / "export"

    with patch(
        "lorcana_training.pretrain.export.download_cards",
        return_value=(tmp_path / "cards.json", cs),
    ):
        result = export_card_embeddings(
            ExportOptions(checkpoint_dir=checkpoint_dir, out_dir=out_dir, device="cpu")
        )

    assert result.card_count == 8
    assert result.embedding_shape == (9, cfg.encoder_dim)  # PAD + 8 cards

    for name in (
        "card_embeddings.fp32.safetensors",
        "encoder_weights.safetensors",
        "tokeniser.json",
        "encoder-manifest.json",
    ):
        assert (out_dir / name).exists(), name

    emb = load_numpy_file(str(out_dir / "card_embeddings.fp32.safetensors"))["card_embeddings"]
    assert emb.shape == (9, cfg.encoder_dim)
    assert emb.dtype == np.float32
    # Row 0 (PAD) is all zeros.
    assert np.allclose(emb[0], 0.0)
    # Real rows are L2-normalised unit vectors (encoder defaults).
    norms = np.linalg.norm(emb[1:], axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    manifest = json.loads((out_dir / "encoder-manifest.json").read_text())
    assert manifest["embedding"]["rows"] == 9
    assert manifest["embedding"]["dim"] == cfg.encoder_dim
    assert manifest["embedding"]["sha256"].startswith("sha256:")
    assert manifest["sources"]["cardsReleaseTag"]
    assert manifest["sources"]["prepareContentHash"] == "sha256:fake"
    assert manifest["sources"]["pretrainBestEpoch"] == 1


def test_export_rejects_tokeniser_vocab_mismatch(tmp_path: Path) -> None:
    cs = _fake_card_set()
    checkpoint_dir, _ = _build_fake_checkpoint(tmp_path / "ckpt", cs)
    # Tamper the checkpoint's stored vocab_size so it disagrees with
    # the saved tokeniser.json — the precise failure mode the guard
    # exists to catch ("someone retrained the tokeniser after the
    # encoder was checkpointed").
    ckpt = torch.load(checkpoint_dir / "encoder.pt", map_location="cpu", weights_only=False)
    ckpt["encoder_config"]["vocab_size"] = ckpt["encoder_config"]["vocab_size"] + 1
    torch.save(ckpt, checkpoint_dir / "encoder.pt")

    with (
        patch(
            "lorcana_training.pretrain.export.download_cards",
            return_value=(tmp_path / "cards.json", cs),
        ),
        pytest.raises(ValueError, match="tokeniser vocab_size"),
    ):
        export_card_embeddings(
            ExportOptions(checkpoint_dir=checkpoint_dir, out_dir=tmp_path / "export", device="cpu")
        )


@pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1",
    reason="set RUN_NETWORK_TESTS=1 to enable (uses real cards-vN)",
)
def test_export_live_pinned(tmp_path: Path) -> None:
    # Requires a real checkpoint under ./artifacts/encoder, typically
    # produced by a prior `lorcana-train pretrain-encoder`. Smokes the
    # export end-to-end against the pinned cards release.
    from lorcana_training.config import REPO_ROOT

    ckpt_dir = REPO_ROOT / "artifacts" / "encoder"
    if not (ckpt_dir / "encoder.pt").exists():
        pytest.skip("no pretrain checkpoint available; run pretrain-encoder first")
    result = export_card_embeddings(
        ExportOptions(checkpoint_dir=ckpt_dir, out_dir=tmp_path / "export")
    )
    assert result.card_count > 1000
    assert result.embedding_shape[1] == 256  # DESIGN default
