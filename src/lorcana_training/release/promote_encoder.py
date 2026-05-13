"""CLI: ``lorcana-train promote-encoder``.

Stages an export bundle, validates it against the pinned ``cards-vN``,
and publishes it as an ``encoder-v<version>`` GitHub release.

Mirrors the shape of ``pnpm tournaments:promote`` in lorcana-scraper:

  1. Copy the four exported artifacts into a clean staging dir so the
     pretrain/export working directories stay untouched.
  2. Re-verify each asset's sha256 against the manifest. Fails if the
     bundle has been tampered with since export.
  3. Confirm the encoder's cardsReleaseTag matches
     ``config/training.yaml`` — i.e. the release we're about to
     publish was trained against the pinned cards, not a stale copy.
  4. Generate release notes (auto-derived from the manifest).
  5. Call ``gh release create`` (skippable via ``--dry-run``).

Usage:

    uv run lorcana-train promote-encoder \\
        --from ./artifacts/encoder-export \\
        --version 0.1.0 \\
        --prerelease

Every release is a prerelease by default because encoder-v0.x runs
are exploratory; bump to ``--no-prerelease`` once an encoder meets
the quality gates we'll add later.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click

from ..config import REPO_ROOT, load_config


@dataclass(frozen=True, slots=True)
class PromoteOptions:
    from_dir: Path
    out_dir: Path
    version: str
    repo: str
    prerelease: bool
    dry_run: bool
    title: str | None
    extra_note: str | None


@dataclass(frozen=True, slots=True)
class PromoteResult:
    out_dir: Path
    tag: str
    manifest_path: Path


_EXPECTED_ASSETS = (
    "card_embeddings.fp32.safetensors",
    "encoder_weights.safetensors",
    "tokeniser.json",
    "encoder-manifest.json",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _stage_bundle(args: PromoteOptions) -> Path:
    """Copy the export bundle into ``out_dir`` and sanity-check assets.

    Keeping the staging dir separate from ``--from`` means the original
    working directory stays intact; callers can re-export and re-promote
    without races.
    """
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in _EXPECTED_ASSETS:
        src = args.from_dir / name
        if not src.exists():
            raise FileNotFoundError(f"expected {name} in {args.from_dir}; not found")
        shutil.copyfile(src, out_dir / name)
    return out_dir


def _verify_manifest(staged_dir: Path) -> dict:
    """Re-hash each asset and compare against ``encoder-manifest.json``."""
    manifest = json.loads((staged_dir / "encoder-manifest.json").read_text(encoding="utf8"))

    pairs = [
        ("card_embeddings.fp32.safetensors", manifest["embedding"]["sha256"]),
        ("encoder_weights.safetensors", manifest["encoderWeights"]["sha256"]),
        ("tokeniser.json", manifest["tokeniser"]["sha256"]),
    ]
    for name, expected in pairs:
        got = _sha256(staged_dir / name)
        if got != expected:
            raise ValueError(
                f"{name}: sha256 mismatch vs encoder-manifest.json "
                f"(got {got}, expected {expected}). Re-export before promoting."
            )
    return manifest


def _check_cards_pin_matches_config(manifest: dict) -> None:
    cfg = load_config()
    tag_in_manifest = manifest.get("sources", {}).get("cardsReleaseTag")
    if tag_in_manifest != cfg.cards_release_tag:
        raise ValueError(
            "encoder was trained against cards release "
            f"{tag_in_manifest!r} but config/training.yaml pins "
            f"{cfg.cards_release_tag!r}. Retrain or bump the pin."
        )


def _build_notes(args: PromoteOptions, manifest: dict) -> str:
    tag = f"encoder-v{args.version}"
    sources = manifest.get("sources", {})
    lines = [f"## {tag}{' (preview)' if args.prerelease else ''}", ""]
    if args.extra_note:
        lines += [args.extra_note, ""]
    lines += [
        f"- **Cards release:** `{sources.get('cardsReleaseTag')}`",
        f"- **`cardSetVersion`:** `{sources.get('cardSetVersion')}`",
        f"- **Pretrain best epoch:** {sources.get('pretrainBestEpoch')}",
        f"- **Held-out total loss:** {sources.get('pretrainBestHeldoutTotal'):.4f}"
        if sources.get("pretrainBestHeldoutTotal") is not None
        else "",
        "",
        "### Artifacts",
        f"- `card_embeddings.fp32.safetensors` — {manifest['embedding']['rows']} rows × "
        f"{manifest['embedding']['dim']} dim, row 0 = PAD",
        "- `encoder_weights.safetensors` — full encoder state for re-encoding",
        "- `tokeniser.json` — BPE tokeniser (self-contained with the bundle)",
        "- `encoder-manifest.json` — provenance + sha256 chain",
    ]
    return "\n".join(line for line in lines if line is not None) + "\n"


def _create_release(args: PromoteOptions, staged_dir: Path, notes_path: Path) -> None:
    tag = f"encoder-v{args.version}"
    title = args.title or (f"{tag} (preview)" if args.prerelease else tag)
    cmd = [
        "gh",
        "release",
        "create",
        tag,
        "--repo",
        args.repo,
        "--title",
        title,
        "--notes-file",
        str(notes_path),
    ]
    if args.prerelease:
        cmd.append("--prerelease")
    cmd.extend(str(staged_dir / name) for name in _EXPECTED_ASSETS)

    sys.stderr.write(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"gh release create exited with status {result.returncode}")


def promote_encoder(args: PromoteOptions) -> PromoteResult:
    staged_dir = _stage_bundle(args)
    manifest = _verify_manifest(staged_dir)
    _check_cards_pin_matches_config(manifest)

    notes_path = staged_dir / "release-notes.md"
    notes_path.write_text(_build_notes(args, manifest), encoding="utf8")

    tag = f"encoder-v{args.version}"
    if args.dry_run:
        sys.stderr.write("[promote-encoder] --dry-run set, not calling gh\n")
    else:
        _create_release(args, staged_dir, notes_path)

    return PromoteResult(
        out_dir=staged_dir,
        tag=tag,
        manifest_path=staged_dir / "encoder-manifest.json",
    )


@click.command("promote-encoder")
@click.option(
    "--from",
    "from_dir",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=REPO_ROOT / "artifacts" / "encoder-export",
    show_default=True,
    help="Directory produced by `lorcana-train export-encoder`.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Staging dir. Defaults to ./artifacts/encoder-v<version>.",
)
@click.option("--version", required=True, help="Semver (e.g. 0.1.0). Becomes encoder-v<version>.")
@click.option("--repo", default="bjorvack/lorcana-training", show_default=True)
@click.option("--prerelease/--no-prerelease", default=True, show_default=True)
@click.option("--dry-run", is_flag=True, help="Stage + verify; skip gh release create.")
@click.option("--title", default=None, help="Override the GitHub release title.")
@click.option(
    "--note", "extra_note", default=None, help="Extra line above the auto-generated body."
)
def promote_encoder_cmd(
    from_dir: Path,
    out_dir: Path | None,
    version: str,
    repo: str,
    prerelease: bool,
    dry_run: bool,
    title: str | None,
    extra_note: str | None,
) -> None:
    """Publish an export bundle as an ``encoder-v<version>`` GitHub release."""
    args = PromoteOptions(
        from_dir=from_dir,
        out_dir=out_dir or REPO_ROOT / "artifacts" / f"encoder-v{version}",
        version=version,
        repo=repo,
        prerelease=prerelease,
        dry_run=dry_run,
        title=title,
        extra_note=extra_note,
    )
    result = promote_encoder(args)
    if args.dry_run:
        click.echo(f"promote-encoder: staged {result.tag} at {result.out_dir}")
    else:
        click.echo(
            f"promote-encoder: published https://github.com/{args.repo}/releases/tag/{result.tag}"
        )
