"""End-to-end smoke test for :func:`lorcana_training.evaluator.train_evaluator`.

Mirrors ``test_proposal_train.py`` — builds a tiny synthetic
``prepared/`` + ``encoder-export/`` on disk, runs 3 epochs (one per
curriculum phase, ``*_epochs=1``) and checks that the loop produced
the expected output artifacts and recorded per-phase history.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lorcana_training.evaluator import EvaluatorOptions, train_evaluator


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _feature_schema(vocab_size: int) -> dict:
    return {
        "dim": 18,
        "slices": {"cost": [0, 12], "inks": [12, 6], "types": [0, 0]},
        "scalars": {},
        "classes": {"inks": ["amber", "amethyst", "emerald", "ruby", "sapphire", "steel"]},
        "normalisers": {},
    }


def _build_prepared(
    tmp_path: Path,
    *,
    vocab_size: int,
    cards_release_tag: str = "cards-vTEST.0.0",
) -> Path:
    prepared = tmp_path / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    (prepared / "vocab.json").write_text(
        json.dumps(
            {
                "padIndex": 0,
                "size": vocab_size,
                "cardSetVersion": "sha256:deadbeef",
                "cardsReleaseTag": cards_release_tag,
            }
        ),
        encoding="utf8",
    )
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "contentHash": "sha256:aaaa",
                "sources": {
                    "cardsReleaseTag": cards_release_tag,
                    "cardSetVersion": "sha256:deadbeef",
                },
            }
        ),
        encoding="utf8",
    )

    # Feature schema + features: every card is cost=1, half amber half ruby.
    schema = _feature_schema(vocab_size)
    (prepared / "feature_schema.json").write_text(json.dumps(schema), encoding="utf8")
    features = torch.zeros(vocab_size + 1, schema["dim"], dtype=torch.float32)
    for i in range(1, vocab_size + 1):
        features[i, 1] = 1.0  # cost=1
        features[i, 12 if i % 2 == 0 else 15] = 1.0  # alternate amber/ruby
    from safetensors.torch import save_file as save_st

    save_st({"card_features": features}, str(prepared / "card_features.safetensors"))

    # Decks: small, in the two inks so the sampler always has a pool.
    def deck(seed: int, inks: list[str]) -> dict:
        base = 1 + (seed % max(1, vocab_size // 2))
        return {
            "cards": [[base, 4], [base + 1, 4], [base + 2, 4]],
            "inks": inks,
        }

    train_decks = [deck(i, ["amber", "ruby"]) for i in range(6)]
    heldout_decks = [deck(i + 50, ["amber", "ruby"]) for i in range(3)]
    _write_jsonl(prepared / "train.evaluator.jsonl", train_decks)
    _write_jsonl(prepared / "heldout.jsonl", heldout_decks)
    return prepared


def _build_encoder_export(
    tmp_path: Path,
    *,
    vocab_size: int,
    embed_dim: int,
    cards_release_tag: str = "cards-vTEST.0.0",
) -> Path:
    export = tmp_path / "encoder-export"
    export.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(7)
    embeddings = torch.randn(vocab_size + 1, embed_dim, generator=g).numpy().astype("float32")
    from safetensors.numpy import save_file

    save_file({"card_embeddings": embeddings}, str(export / "card_embeddings.fp32.safetensors"))
    (export / "encoder-manifest.json").write_text(
        json.dumps(
            {
                "sources": {
                    "cardsReleaseTag": cards_release_tag,
                    "cardSetVersion": "sha256:deadbeef",
                },
                "embedding": {"rows": vocab_size + 1, "dim": embed_dim, "dtype": "float32"},
            }
        ),
        encoding="utf8",
    )
    return export


def test_train_evaluator_smoke(tmp_path: Path) -> None:
    vocab_size = 8
    embed_dim = 16
    prepared = _build_prepared(tmp_path, vocab_size=vocab_size)
    encoder_export = _build_encoder_export(tmp_path, vocab_size=vocab_size, embed_dim=embed_dim)
    opts = EvaluatorOptions(
        prepared_dir=prepared,
        encoder_export_dir=encoder_export,
        out_dir=tmp_path / "evaluator",
        warmup_epochs=1,
        curve_epochs=1,
        local_epochs=1,
        batch_size=4,
        samples_per_deck=2,
        patience=10,
        d_model=embed_dim,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        dropout=0.0,
        device="cpu",
        seed=0,
    )
    result = train_evaluator(opts)

    out = tmp_path / "evaluator"
    assert (out / "evaluator.pt").exists()
    assert (out / "evaluator-run.json").exists()
    assert (out / "evaluator-manifest.json").exists()

    history = json.loads((out / "evaluator-run.json").read_text())["history"]
    assert len(history) == 3
    phases = [h["phase"] for h in history]
    assert phases == ["random_in_ink", "curve_matched", "local_swap"]

    manifest = json.loads((out / "evaluator-manifest.json").read_text())
    assert manifest["bestEpoch"] >= 1
    assert manifest["gradientParameterCount"] > 0
    assert manifest["sources"]["cardsReleaseTag"] == "cards-vTEST.0.0"
    assert result.gradient_parameter_count == manifest["gradientParameterCount"]


def test_train_evaluator_rejects_mismatched_cards_tag(tmp_path: Path) -> None:
    prepared = _build_prepared(tmp_path, vocab_size=8)
    encoder_export = _build_encoder_export(
        tmp_path, vocab_size=8, embed_dim=16, cards_release_tag="cards-vOTHER"
    )
    opts = EvaluatorOptions(
        prepared_dir=prepared,
        encoder_export_dir=encoder_export,
        out_dir=tmp_path / "evaluator",
        warmup_epochs=1,
        curve_epochs=0,
        local_epochs=0,
        batch_size=2,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        dropout=0.0,
        device="cpu",
    )
    with pytest.raises(ValueError, match="Cards-tag mismatch"):
        train_evaluator(opts)
