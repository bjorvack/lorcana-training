"""Unit tests for ``lorcana-train promote-model``.

Covers staging + sha256 verification + manifest cross-checks + dry-run
behaviour. The ``gh release create`` call itself is never exercised
here — any unit test that invoked ``gh`` would either need the real
CLI installed + a token or a brittle mock; the CLI calling a shell
tool is simple enough to eyeball.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lorcana_training.config import TrainingConfig
from lorcana_training.release.promote_model import (
    PromoteModelOptions,
    promote_model,
)


def _stub_config(cards_tag: str):  # type: ignore[no-untyped-def]
    """Return a zero-arg callable that plays the role of ``load_config``.

    Tests patch ``promote_model.load_config`` so the promote flow's
    cards-pin check reads a deterministic config without touching the
    real ``config/training.yaml`` on disk.
    """

    def _fn() -> TrainingConfig:
        return TrainingConfig(
            raw={
                "schemas_repo": "bjorvack/lorcana-schemas",
                "schemas_release_tag": "v0.0.0",
                "scraper_repo": "bjorvack/lorcana-scraper",
                "cards_release_tag": cards_tag,
                "tournaments_release_tag": "tournaments-vTEST",
            }
        )

    return _fn


def _write_file(path: Path, content: bytes) -> str:
    import hashlib

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    h = hashlib.sha256()
    h.update(content)
    return "sha256:" + h.hexdigest()


def _build_bundle(tmp_path: Path, *, cards_tag: str = "cards-vTEST") -> dict[str, Path]:
    """Lay out a realistic-shape export + tables + prepared triple on disk."""
    export = tmp_path / "export"
    tables = tmp_path / "tables"
    prepared = tmp_path / "prepared"

    # --- ONNX side ---
    p_onnx = _write_file(export / "proposal.onnx", b"proposal-onnx-bytes")
    p_data = _write_file(export / "proposal.onnx.data", b"proposal-data-bytes")
    e_onnx = _write_file(export / "evaluator.onnx", b"evaluator-onnx-bytes")
    e_data = _write_file(export / "evaluator.onnx.data", b"evaluator-data-bytes")
    emb_sha = _write_file(export / "card_embeddings.bin", b"\x01\x02\x03\x04")
    encoder_manifest_stub = {
        "embedding": {"rows": 5, "dim": 4, "dtype": "float16", "sha256": "sha256:enc-emb"},
    }
    export_manifest = {
        "generatedAt": "2026-05-13T00:00:00+00:00",
        "opset": 17,
        "proposal": {
            "path": "proposal.onnx",
            "sha256": p_onnx,
            "externalData": {
                "path": "proposal.onnx.data",
                "sha256": p_data,
                "bytes": len(b"proposal-data-bytes"),
            },
            "inputNames": ["card_ids", "ink_multihot", "card_embeddings"],
            "outputNames": ["logits"],
        },
        "evaluator": {
            "path": "evaluator.onnx",
            "sha256": e_onnx,
            "externalData": {
                "path": "evaluator.onnx.data",
                "sha256": e_data,
                "bytes": len(b"evaluator-data-bytes"),
            },
            "inputNames": ["partial_ids", "candidate_ids", "card_embeddings"],
            "outputNames": ["logits"],
        },
        "cardEmbeddings": {
            "path": "card_embeddings.bin",
            "sha256": emb_sha,
            "rows": 5,
            "dim": 4,
            "dtype": "float16",
            "padRow": 0,
        },
        "vocabSize": 4,
        "sources": {
            "cardsReleaseTag": cards_tag,
            "cardSetVersion": "sha256:test",
            "encoderManifest": encoder_manifest_stub,
            "proposalManifest": {"bestEpoch": 7},
            "evaluatorManifest": {"bestEpoch": 9},
        },
    }
    (export / "export-manifest.json").write_text(json.dumps(export_manifest), encoding="utf8")

    # --- Tables side ---
    freq_sha = _write_file(tables / "play_frequency.json", b'{"_all": {"1": 0.5}}')
    cent_sha = _write_file(tables / "archetype_centroids.json", b'{"k": 2}')
    tables_manifest = {
        "generatedAt": "2026-05-13T00:01:00+00:00",
        "playFrequency": {"path": "play_frequency.json", "sha256": freq_sha, "inkPairCount": 1},
        "archetypeCentroids": {
            "path": "archetype_centroids.json",
            "sha256": cent_sha,
            "k": 20,
        },
        "sources": {
            "cardsReleaseTag": cards_tag,
            "cardSetVersion": "sha256:test",
            "encoderManifest": encoder_manifest_stub,
        },
        "splits": {"trainDecks": 2780},
    }
    (tables / "tables-manifest.json").write_text(json.dumps(tables_manifest), encoding="utf8")

    # --- Prepared side (just vocab.json for the bundle) ---
    (prepared / "vocab.json").parent.mkdir(parents=True, exist_ok=True)
    (prepared / "vocab.json").write_text(
        json.dumps({"padIndex": 0, "size": 4}),
        encoding="utf8",
    )
    return {"export": export, "tables": tables, "prepared": prepared}


def test_promote_model_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # load_config() needs to return the same cards tag the manifests use.
    from lorcana_training import config as cfg_module

    from lorcana_training.release import promote_model as pm_module

    monkeypatch.setattr(pm_module, "load_config", _stub_config("cards-vTEST"))

    dirs = _build_bundle(tmp_path)
    opts = PromoteModelOptions(
        export_dir=dirs["export"],
        tables_dir=dirs["tables"],
        prepared_dir=dirs["prepared"],
        out_dir=tmp_path / "model-v0.0.0",
        version="0.0.0",
        repo="bjorvack/lorcana-training",
        prerelease=True,
        dry_run=True,
        title=None,
        extra_note=None,
    )
    result = promote_model(opts)

    # Staging dir has the full bundle.
    assert (result.out_dir / "proposal.onnx").exists()
    assert (result.out_dir / "proposal.onnx.data").exists()
    assert (result.out_dir / "evaluator.onnx").exists()
    assert (result.out_dir / "evaluator.onnx.data").exists()
    assert (result.out_dir / "card_embeddings.bin").exists()
    assert (result.out_dir / "play_frequency.json").exists()
    assert (result.out_dir / "archetype_centroids.json").exists()
    assert (result.out_dir / "vocab.json").exists()
    assert (result.out_dir / "release-notes.md").exists()

    # Unified manifest has the right structure.
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["tag"] == "model-v0.0.0"
    assert manifest["prerelease"] is True
    assert manifest["vocabSize"] == 4
    assert manifest["opset"] == 17
    assert manifest["sources"]["cardsReleaseTag"] == "cards-vTEST"
    # All core assets recorded with sha256.
    for key in (
        "proposal",
        "evaluator",
        "cardEmbeddings",
        "playFrequency",
        "archetypeCentroids",
        "vocab",
    ):
        assert manifest["assets"][key]["sha256"].startswith("sha256:")


def test_promote_model_rejects_cards_tag_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lorcana_training.release import promote_model as pm_module

    monkeypatch.setattr(pm_module, "load_config", _stub_config("cards-vDIFFERENT"))

    dirs = _build_bundle(tmp_path, cards_tag="cards-vTEST")
    opts = PromoteModelOptions(
        export_dir=dirs["export"],
        tables_dir=dirs["tables"],
        prepared_dir=dirs["prepared"],
        out_dir=tmp_path / "staging",
        version="0.0.0",
        repo="bjorvack/lorcana-training",
        prerelease=True,
        dry_run=True,
        title=None,
        extra_note=None,
    )
    with pytest.raises(ValueError, match="cards-vDIFFERENT"):
        promote_model(opts)


def test_promote_model_rejects_sha_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lorcana_training.release import promote_model as pm_module

    monkeypatch.setattr(pm_module, "load_config", _stub_config("cards-vTEST"))

    dirs = _build_bundle(tmp_path)
    # Mutate proposal.onnx after the manifest was sealed. Staging
    # copies the file fresh into staging, so we need to mutate the
    # *source* before promote runs.
    (dirs["export"] / "proposal.onnx").write_bytes(b"tampered-bytes")

    opts = PromoteModelOptions(
        export_dir=dirs["export"],
        tables_dir=dirs["tables"],
        prepared_dir=dirs["prepared"],
        out_dir=tmp_path / "staging",
        version="0.0.0",
        repo="bjorvack/lorcana-training",
        prerelease=True,
        dry_run=True,
        title=None,
        extra_note=None,
    )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        promote_model(opts)


def test_promote_model_wet_run_invokes_gh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Not a --dry-run: verify the staging + manifest pipeline calls
    subprocess.run with ``gh release create``. Skips the real network
    by patching subprocess.run to a fake."""
    from lorcana_training.release import promote_model as pm_module

    monkeypatch.setattr(pm_module, "load_config", _stub_config("cards-vTEST"))

    dirs = _build_bundle(tmp_path)
    opts = PromoteModelOptions(
        export_dir=dirs["export"],
        tables_dir=dirs["tables"],
        prepared_dir=dirs["prepared"],
        out_dir=tmp_path / "staging",
        version="0.0.0",
        repo="bjorvack/lorcana-training",
        prerelease=True,
        dry_run=False,
        title=None,
        extra_note=None,
    )
    with patch.object(pm_module.subprocess, "run") as mock_run:
        mock_run.return_value.returncode = 0
        promote_model(opts)
        assert mock_run.called
        called_cmd = mock_run.call_args.args[0]
        assert called_cmd[0] == "gh"
        assert called_cmd[1] == "release"
        assert called_cmd[2] == "create"
        assert "model-v0.0.0" in called_cmd
