"""End-to-end smoke test for :func:`lorcana_training.proposal.train_proposal`.

Runs a 2-epoch training loop over a hand-crafted ``prepared/`` +
``encoder-export/`` on tiny synthetic data. Confirms that the loop

  - produces the expected output files (checkpoint + run.json +
    manifest),
  - writes a non-trivial history (training loss goes somewhere),
  - refuses mismatched cards-tag provenance.

The real proposal training can take tens of minutes; this test is
pure correctness and should finish in a few seconds on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lorcana_training.proposal import ProposalOptions, train_proposal


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _build_prepared_dir(
    tmp_path: Path,
    *,
    vocab_size: int,
    cards_release_tag: str = "cards-vTEST.0.0",
    n_train: int = 4,
    n_heldout: int = 2,
) -> Path:
    prepared = tmp_path / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    # Minimal vocab.json — only the fields the loader reads.
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
    # Minimal manifest.json with the provenance keys the trainer checks.
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

    # Build a tiny deck stream. 4 cards × 4 copies = 16 total copies,
    # far below 60 — the proposal stage doesn't enforce a minimum, only
    # prepare does. We just need enough positions for the masking to
    # produce meaningful partials.
    def deck(seed: int, inks: list[str]) -> dict:
        base = 1 + (seed % max(1, vocab_size // 2))
        return {
            "cards": [[base, 4], [base + 1, 4], [base + 2, 4], [base + 3, 4]],
            "inks": inks,
        }

    train_decks = [deck(i, ["amber", "ruby"]) for i in range(n_train)]
    heldout_decks = [deck(i + 100, ["amber", "ruby"]) for i in range(n_heldout)]
    # Write BOTH splits so the test exercises the default
    # (train.evaluator.jsonl) without caring whether the default flips
    # again in a future sweep. train.proposal.jsonl is also present so
    # tests that want to exercise the recency-filtered path can switch
    # with just the ``train_split`` option.
    _write_jsonl(prepared / "train.proposal.jsonl", train_decks)
    _write_jsonl(prepared / "train.evaluator.jsonl", train_decks)
    _write_jsonl(prepared / "heldout.jsonl", heldout_decks)
    return prepared


def _build_encoder_export_dir(
    tmp_path: Path,
    *,
    vocab_size: int,
    embed_dim: int,
    cards_release_tag: str = "cards-vTEST.0.0",
) -> Path:
    export = tmp_path / "encoder-export"
    export.mkdir(parents=True, exist_ok=True)
    # Deterministic fake embeddings so the test is reproducible.
    g = torch.Generator().manual_seed(7)
    embeddings = torch.randn(vocab_size + 1, embed_dim, generator=g).numpy().astype("float32")

    from safetensors.numpy import save_file

    save_file(
        {"card_embeddings": embeddings},
        str(export / "card_embeddings.fp32.safetensors"),
    )
    (export / "encoder-manifest.json").write_text(
        json.dumps(
            {
                "sources": {
                    "cardsReleaseTag": cards_release_tag,
                    "cardSetVersion": "sha256:deadbeef",
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
    return export


def test_train_proposal_smoke(tmp_path: Path) -> None:
    vocab_size = 16
    embed_dim = 16
    prepared = _build_prepared_dir(tmp_path, vocab_size=vocab_size)
    encoder_export = _build_encoder_export_dir(tmp_path, vocab_size=vocab_size, embed_dim=embed_dim)
    out_dir = tmp_path / "proposal"

    opts = ProposalOptions(
        prepared_dir=prepared,
        encoder_export_dir=encoder_export,
        out_dir=out_dir,
        epochs=2,
        batch_size=4,
        samples_per_deck=2,
        patience=10,  # don't early-stop during the 2-epoch smoke
        d_model=embed_dim,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        dropout=0.0,
        device="cpu",
        num_workers=0,
        seed=0,
    )
    result = train_proposal(opts)

    # Outputs exist.
    assert (out_dir / "proposal.pt").exists()
    assert (out_dir / "proposal-run.json").exists()
    assert (out_dir / "proposal-manifest.json").exists()

    history = json.loads((out_dir / "proposal-run.json").read_text())["history"]
    assert len(history) == 2
    # Each entry has the metrics we promised.
    for entry in history:
        assert set(entry.keys()) >= {
            "epoch",
            "train_total",
            "train_ce",
            "train_entropy",
            "heldout_total",
            "heldout_ce",
            "heldout_entropy",
            "lr",
            "elapsed_s",
        }

    manifest = json.loads((out_dir / "proposal-manifest.json").read_text())
    assert manifest["bestEpoch"] in (1, 2)
    assert manifest["gradientParameterCount"] > 0
    assert manifest["sources"]["cardsReleaseTag"] == "cards-vTEST.0.0"
    assert manifest["splits"]["trainDecks"] == 4
    assert manifest["splits"]["heldoutDecks"] == 2
    assert result.gradient_parameter_count == manifest["gradientParameterCount"]


def test_train_proposal_rejects_mismatched_cards_tag(tmp_path: Path) -> None:
    vocab_size = 8
    embed_dim = 16
    prepared = _build_prepared_dir(
        tmp_path, vocab_size=vocab_size, cards_release_tag="cards-vTEST.0.0"
    )
    encoder_export = _build_encoder_export_dir(
        tmp_path,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        cards_release_tag="cards-vOTHER.0.0",
    )

    opts = ProposalOptions(
        prepared_dir=prepared,
        encoder_export_dir=encoder_export,
        out_dir=tmp_path / "proposal",
        epochs=1,
        batch_size=2,
        d_model=embed_dim,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        dropout=0.0,
        device="cpu",
        num_workers=0,
    )
    with pytest.raises(ValueError, match="Cards-tag mismatch"):
        train_proposal(opts)


def test_train_proposal_rejects_missing_prepare_manifest(tmp_path: Path) -> None:
    (tmp_path / "prepared").mkdir()  # empty dir, no manifest
    opts = ProposalOptions(
        prepared_dir=tmp_path / "prepared",
        encoder_export_dir=tmp_path,  # unused before the manifest check
        out_dir=tmp_path / "proposal",
    )
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        train_proposal(opts)
