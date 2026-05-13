"""CLI: ``lorcana-train promote-model``.

Bundles the ONNX export + tables into a ``model-v<version>`` GitHub
release. Parallel in structure to :mod:`promote_encoder`:

  1. Copy the full set of artifacts (ONNX graphs + sidecar data +
     card_embeddings.bin + play_frequency + archetype_centroids +
     vocab.json) into a clean staging dir.
  2. Re-verify each asset's sha256 against its source manifest
     (``export-manifest.json`` for the ONNX side,
     ``tables-manifest.json`` for the tables side). Fails if
     anything was tampered with between export and promote.
  3. Cross-check the cards/encoder pins: every upstream manifest
     must point to the same cards-vN and cardSetVersion; otherwise
     the bundle would mix neural nets trained against drifted
     vocabularies.
  4. Build a unified ``model-manifest.json`` that the web client
     loads first and uses to drive fetches of the sibling assets.
  5. Write Markdown release notes from the combined metadata.
  6. Call ``gh release create`` unless ``--dry-run``.

Usage:

    uv run lorcana-train promote-model \\
        --export ./artifacts/model-export \\
        --tables ./artifacts/tables \\
        --version 0.1.0 \\
        --prerelease

``--prerelease`` is the default because v0.x models haven't been
through the (not-yet-written) quality-gate gauntlet. Flip it to
``--no-prerelease`` once the eval stage is in place.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from ..config import REPO_ROOT, load_config


# Asset names that land verbatim in the staging dir. The ``.data``
# sidecars are treated specially because the dynamo exporter emits
# them unconditionally and they must ship alongside the ``.onnx`` graph.
_ONNX_ASSETS = (
    "proposal.onnx",
    "evaluator.onnx",
    "card_embeddings.bin",
)
_TABLE_ASSETS = (
    "play_frequency.json",
    "archetype_centroids.json",
)


@dataclass(frozen=True, slots=True)
class PromoteModelOptions:
    export_dir: Path
    tables_dir: Path
    prepared_dir: Path
    out_dir: Path
    version: str
    repo: str
    prerelease: bool
    dry_run: bool
    title: str | None
    extra_note: str | None


@dataclass(frozen=True, slots=True)
class PromoteModelResult:
    out_dir: Path
    tag: str
    manifest_path: Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf8"))
    return payload


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    shutil.copyfile(src, dst)
    return True


def _stage_bundle(opts: PromoteModelOptions) -> Path:
    out_dir = opts.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in _ONNX_ASSETS:
        src = opts.export_dir / name
        if not src.exists():
            raise FileNotFoundError(f"expected {name} in {opts.export_dir}; not found")
        shutil.copyfile(src, out_dir / name)
        # Copy matching sidecar if the dynamo exporter produced one.
        _copy_if_exists(
            src.with_name(src.name + ".data"), (out_dir / name).with_name(name + ".data")
        )

    for name in _TABLE_ASSETS:
        src = opts.tables_dir / name
        if not src.exists():
            raise FileNotFoundError(f"expected {name} in {opts.tables_dir}; not found")
        shutil.copyfile(src, out_dir / name)

    # vocab.json comes from prepare, not either export. The web
    # client needs it to go from logical card id back to the
    # canonical printing the user sees, so bundling it with the
    # model keeps the release self-contained.
    vocab_src = opts.prepared_dir / "vocab.json"
    if not vocab_src.exists():
        raise FileNotFoundError(f"expected vocab.json in {opts.prepared_dir}; not found")
    shutil.copyfile(vocab_src, out_dir / "vocab.json")
    return out_dir


def _verify_export_manifest(staged_dir: Path, export_dir: Path) -> dict[str, Any]:
    """Re-hash each export asset and compare against ``export-manifest.json``."""
    export_manifest_src = export_dir / "export-manifest.json"
    if not export_manifest_src.exists():
        raise FileNotFoundError(f"expected export-manifest.json in {export_dir}; not found")
    manifest = _load_json(export_manifest_src)

    pairs: list[tuple[str, str]] = [
        ("proposal.onnx", manifest["proposal"]["sha256"]),
        ("evaluator.onnx", manifest["evaluator"]["sha256"]),
        ("card_embeddings.bin", manifest["cardEmbeddings"]["sha256"]),
    ]
    for name, expected in pairs:
        got = _sha256(staged_dir / name)
        if got != expected:
            raise ValueError(
                f"{name}: sha256 mismatch vs export-manifest.json "
                f"(got {got}, expected {expected}). Re-export before promoting.",
            )
    # Also check .data sidecars when the manifest lists them.
    for model_key, sidecar_name in (
        ("proposal", "proposal.onnx.data"),
        ("evaluator", "evaluator.onnx.data"),
    ):
        sidecar_info = manifest[model_key].get("externalData")
        if sidecar_info is None:
            continue
        got = _sha256(staged_dir / sidecar_name)
        if got != sidecar_info["sha256"]:
            raise ValueError(
                f"{sidecar_name}: sha256 mismatch vs export-manifest.json "
                f"(got {got}, expected {sidecar_info['sha256']}).",
            )
    return manifest


def _verify_tables_manifest(staged_dir: Path, tables_dir: Path) -> dict[str, Any]:
    tables_manifest_src = tables_dir / "tables-manifest.json"
    if not tables_manifest_src.exists():
        raise FileNotFoundError(f"expected tables-manifest.json in {tables_dir}; not found")
    manifest = _load_json(tables_manifest_src)
    for name, key in (
        ("play_frequency.json", "playFrequency"),
        ("archetype_centroids.json", "archetypeCentroids"),
    ):
        expected = manifest[key]["sha256"]
        got = _sha256(staged_dir / name)
        if got != expected:
            raise ValueError(
                f"{name}: sha256 mismatch vs tables-manifest.json "
                f"(got {got}, expected {expected}).",
            )
    return manifest


def _check_pins_consistent(
    *,
    export_manifest: dict[str, Any],
    tables_manifest: dict[str, Any],
) -> None:
    """All three upstream manifests (export, tables, config) must agree
    on the cards release tag. If a caller re-ran prepare against a
    different cards-vN between training stages, the ONNX graphs and
    the frequency tables would talk past each other."""
    cfg = load_config()
    export_cards = export_manifest["sources"].get("cardsReleaseTag")
    tables_cards = tables_manifest["sources"].get("cardsReleaseTag")
    if export_cards != cfg.cards_release_tag:
        raise ValueError(
            f"export manifest pins {export_cards!r} but config "
            f"pins {cfg.cards_release_tag!r}. Re-export or bump the pin.",
        )
    if tables_cards != cfg.cards_release_tag:
        raise ValueError(
            f"tables manifest pins {tables_cards!r} but config "
            f"pins {cfg.cards_release_tag!r}. Re-build tables or bump the pin.",
        )
    # Additional: encoder-export sha256 must be the same in both
    # manifests so we know both stages consumed the same embeddings.
    export_encoder = (
        export_manifest["sources"]["encoderManifest"].get("embedding", {}).get("sha256")
    )
    tables_encoder = (
        tables_manifest["sources"]["encoderManifest"].get("embedding", {}).get("sha256")
    )
    if export_encoder and tables_encoder and export_encoder != tables_encoder:
        raise ValueError(
            f"encoder-embedding sha256 drift: export {export_encoder} vs "
            f"tables {tables_encoder}. One of them ran against a stale "
            "encoder-export; re-run both before promoting.",
        )


def _build_model_manifest(
    staged_dir: Path,
    *,
    opts: PromoteModelOptions,
    export_manifest: dict[str, Any],
    tables_manifest: dict[str, Any],
) -> Path:
    cfg = load_config()
    tag = f"model-v{opts.version}"

    def _asset(name: str) -> dict[str, Any]:
        path = staged_dir / name
        return {"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)}

    manifest: dict[str, Any] = {
        "tag": tag,
        "version": opts.version,
        "generatedAt": export_manifest.get("generatedAt"),
        "prerelease": opts.prerelease,
        "vocabSize": export_manifest["vocabSize"],
        "opset": export_manifest["opset"],
        "sources": {
            "repo": opts.repo,
            "cardsReleaseTag": cfg.cards_release_tag,
            "cardSetVersion": export_manifest["sources"].get("cardSetVersion"),
            "tournamentsReleaseTag": cfg.tournaments_release_tag,
            "schemasReleaseTag": cfg.schemas_release_tag,
            "encoderManifest": export_manifest["sources"].get("encoderManifest"),
            "proposalManifest": export_manifest["sources"].get("proposalManifest"),
            "evaluatorManifest": export_manifest["sources"].get("evaluatorManifest"),
            "tablesManifest": {
                "generatedAt": tables_manifest.get("generatedAt"),
                "kArchetypes": tables_manifest["archetypeCentroids"].get("k"),
                "inkPairCount": tables_manifest["playFrequency"].get("inkPairCount"),
                "trainDecks": tables_manifest.get("splits", {}).get("trainDecks"),
            },
        },
        "assets": {
            "proposal": {
                **_asset("proposal.onnx"),
                "externalData": _asset("proposal.onnx.data")
                if (staged_dir / "proposal.onnx.data").exists()
                else None,
                "inputNames": export_manifest["proposal"]["inputNames"],
                "outputNames": export_manifest["proposal"]["outputNames"],
            },
            "evaluator": {
                **_asset("evaluator.onnx"),
                "externalData": _asset("evaluator.onnx.data")
                if (staged_dir / "evaluator.onnx.data").exists()
                else None,
                "inputNames": export_manifest["evaluator"]["inputNames"],
                "outputNames": export_manifest["evaluator"]["outputNames"],
            },
            "cardEmbeddings": {
                **_asset("card_embeddings.bin"),
                "rows": export_manifest["cardEmbeddings"]["rows"],
                "dim": export_manifest["cardEmbeddings"]["dim"],
                "dtype": export_manifest["cardEmbeddings"]["dtype"],
                "padRow": export_manifest["cardEmbeddings"]["padRow"],
            },
            "playFrequency": _asset("play_frequency.json"),
            "archetypeCentroids": _asset("archetype_centroids.json"),
            "vocab": _asset("vocab.json"),
        },
    }
    manifest_path = staged_dir / "model-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    return manifest_path


def _build_notes(
    opts: PromoteModelOptions,
    *,
    model_manifest: dict[str, Any],
    export_manifest: dict[str, Any],
    tables_manifest: dict[str, Any],
) -> str:
    tag = model_manifest["tag"]
    lines = [f"## {tag}{' (preview)' if opts.prerelease else ''}", ""]
    if opts.extra_note:
        lines += [opts.extra_note, ""]
    sources = model_manifest["sources"]
    lines += [
        f"- **Cards release:** `{sources['cardsReleaseTag']}`",
        f"- **Tournaments release:** `{sources['tournamentsReleaseTag']}`",
        f"- **Schemas release:** `{sources['schemasReleaseTag']}`",
        f"- **cardSetVersion:** `{sources['cardSetVersion']}`",
        f"- **ONNX opset:** {model_manifest['opset']}",
        f"- **Vocab size:** {model_manifest['vocabSize']}",
        f"- **Train decks (tables):** {sources['tablesManifest']['trainDecks']}",
        f"- **Archetype k:** {sources['tablesManifest']['kArchetypes']}",
        "",
        "### Assets",
        f"- `proposal.onnx` ({_fmt_bytes(model_manifest['assets']['proposal']['bytes'])})"
        + _sidecar_line(model_manifest["assets"]["proposal"].get("externalData")),
        f"- `evaluator.onnx` ({_fmt_bytes(model_manifest['assets']['evaluator']['bytes'])})"
        + _sidecar_line(model_manifest["assets"]["evaluator"].get("externalData")),
        f"- `card_embeddings.bin` ({_fmt_bytes(model_manifest['assets']['cardEmbeddings']['bytes'])}, "
        f"{model_manifest['assets']['cardEmbeddings']['rows']} × "
        f"{model_manifest['assets']['cardEmbeddings']['dim']} "
        f"{model_manifest['assets']['cardEmbeddings']['dtype']})",
        f"- `play_frequency.json` ({_fmt_bytes(model_manifest['assets']['playFrequency']['bytes'])})",
        f"- `archetype_centroids.json` ({_fmt_bytes(model_manifest['assets']['archetypeCentroids']['bytes'])})",
        f"- `vocab.json` ({_fmt_bytes(model_manifest['assets']['vocab']['bytes'])})",
        "- `model-manifest.json` — provenance + sha256 chain",
    ]
    _ = export_manifest
    _ = tables_manifest
    return "\n".join(lines) + "\n"


def _sidecar_line(sidecar: dict[str, Any] | None) -> str:
    if sidecar is None:
        return ""
    return f" + `{sidecar['path']}` ({_fmt_bytes(sidecar['bytes'])})"


def _fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(n)
    for u in units:
        if size < 1024:
            return f"{size:.1f} {u}"
        size /= 1024
    return f"{size:.1f} TB"


def _release_assets(staged_dir: Path) -> list[str]:
    """The exact set of files the ``gh release create`` should attach.

    Ordering is stable so `gh release view` lists assets
    consistently and the release notes + actual file list agree on
    what the bundle contains.
    """
    files = [
        "proposal.onnx",
        "evaluator.onnx",
        "card_embeddings.bin",
        "play_frequency.json",
        "archetype_centroids.json",
        "vocab.json",
        "model-manifest.json",
    ]
    # Optional sidecars live next to the .onnx they belong to.
    for sidecar in ("proposal.onnx.data", "evaluator.onnx.data"):
        if (staged_dir / sidecar).exists():
            # Insert right after the corresponding .onnx for
            # readability in the `gh` output.
            base = sidecar.rsplit(".data", 1)[0]
            index = files.index(base)
            files.insert(index + 1, sidecar)
    return [str(staged_dir / f) for f in files]


def _create_release(
    opts: PromoteModelOptions,
    staged_dir: Path,
    notes_path: Path,
    tag: str,
) -> None:
    title = opts.title or (f"{tag} (preview)" if opts.prerelease else tag)
    cmd = [
        "gh",
        "release",
        "create",
        tag,
        "--repo",
        opts.repo,
        "--title",
        title,
        "--notes-file",
        str(notes_path),
    ]
    if opts.prerelease:
        cmd.append("--prerelease")
    cmd.extend(_release_assets(staged_dir))
    sys.stderr.write(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"gh release create exited with status {result.returncode}")


def promote_model(opts: PromoteModelOptions) -> PromoteModelResult:
    staged_dir = _stage_bundle(opts)
    export_manifest = _verify_export_manifest(staged_dir, opts.export_dir)
    tables_manifest = _verify_tables_manifest(staged_dir, opts.tables_dir)
    _check_pins_consistent(export_manifest=export_manifest, tables_manifest=tables_manifest)
    manifest_path = _build_model_manifest(
        staged_dir,
        opts=opts,
        export_manifest=export_manifest,
        tables_manifest=tables_manifest,
    )
    model_manifest = _load_json(manifest_path)

    notes_path = staged_dir / "release-notes.md"
    notes_path.write_text(
        _build_notes(
            opts,
            model_manifest=model_manifest,
            export_manifest=export_manifest,
            tables_manifest=tables_manifest,
        ),
        encoding="utf8",
    )

    tag = model_manifest["tag"]
    if opts.dry_run:
        sys.stderr.write("[promote-model] --dry-run set, not calling gh\n")
    else:
        _create_release(opts, staged_dir, notes_path, tag)

    return PromoteModelResult(out_dir=staged_dir, tag=tag, manifest_path=manifest_path)


@click.command("promote-model")
@click.option(
    "--export",
    "export_dir",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=REPO_ROOT / "artifacts" / "model-export",
    show_default=True,
)
@click.option(
    "--tables",
    "tables_dir",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=REPO_ROOT / "artifacts" / "tables",
    show_default=True,
)
@click.option(
    "--prepared",
    "prepared_dir",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=REPO_ROOT / "prepared",
    show_default=True,
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Staging dir. Defaults to ./artifacts/model-v<version>.",
)
@click.option("--version", required=True, help="Semver. Becomes model-v<version>.")
@click.option("--repo", default="bjorvack/lorcana-training", show_default=True)
@click.option("--prerelease/--no-prerelease", default=True, show_default=True)
@click.option("--dry-run", is_flag=True, help="Stage + verify; skip gh release create.")
@click.option("--title", default=None, help="Override the GitHub release title.")
@click.option(
    "--note", "extra_note", default=None, help="Extra line above the auto-generated body."
)
def promote_model_cmd(
    export_dir: Path,
    tables_dir: Path,
    prepared_dir: Path,
    out_dir: Path | None,
    version: str,
    repo: str,
    prerelease: bool,
    dry_run: bool,
    title: str | None,
    extra_note: str | None,
) -> None:
    """Stage, verify, and publish a ``model-v<version>`` GitHub release."""
    opts = PromoteModelOptions(
        export_dir=export_dir,
        tables_dir=tables_dir,
        prepared_dir=prepared_dir,
        out_dir=out_dir or REPO_ROOT / "artifacts" / f"model-v{version}",
        version=version,
        repo=repo,
        prerelease=prerelease,
        dry_run=dry_run,
        title=title,
        extra_note=extra_note,
    )
    result = promote_model(opts)
    if opts.dry_run:
        click.echo(f"promote-model: staged {result.tag} at {result.out_dir}")
    else:
        click.echo(
            f"promote-model: published https://github.com/{opts.repo}/releases/tag/{result.tag}"
        )
