"""Command-line entry point: ``lorcana-train``."""

from __future__ import annotations

from pathlib import Path

import click

from .config import REPO_ROOT
from .prepare import PrepareOptions, prepare as run_prepare
from .pretrain import (
    ExportOptions,
    PretrainOptions,
    export_card_embeddings as run_export,
    pretrain_encoder as run_pretrain,
)


@click.group()
def main() -> None:
    """Lorcana training CLI."""


@main.command()
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=REPO_ROOT / "prepared",
    show_default=True,
    help="Where to write prepared artifacts.",
)
@click.option(
    "--strict-legality",
    is_flag=True,
    default=False,
    help="Drop decks containing any not_legal card. Default: Infinity-format "
    "(drop only truly banned cards).",
)
@click.option(
    "--proposal-recency-months",
    type=int,
    default=12,
    show_default=True,
    help="Trailing window for the proposal net training split.",
)
@click.option(
    "--heldout-ratio",
    type=float,
    default=0.10,
    show_default=True,
    help="Fraction of validated decks to reserve for held-out eval (stratified).",
)
@click.option("--seed", type=int, default=0, show_default=True, help="Split shuffle seed.")
@click.option("--force", is_flag=True, help="Ignore the cache and rebuild unconditionally.")
def prepare(
    out_dir: Path,
    strict_legality: bool,
    proposal_recency_months: int,
    heldout_ratio: float,
    seed: int,
    force: bool,
) -> None:
    """Download pinned artifacts, build vocab+features, validate, split.

    Writes ``<out_dir>/manifest.json`` + the six artifacts referenced
    from it. Subsequent invocations with the same inputs short-circuit;
    pass ``--force`` to rebuild.
    """
    opts = PrepareOptions(
        out_dir=out_dir,
        strict_legality=strict_legality,
        proposal_recency_months=proposal_recency_months,
        heldout_ratio=heldout_ratio,
        seed=seed,
        force=force,
    )
    result = run_prepare(opts)
    if result.cached:
        click.echo(f"prepare: cache hit ({result.content_hash}). wrote {result.manifest_path}")
    else:
        click.echo(f"prepare: built {result.content_hash}. wrote {result.manifest_path}")


@main.command("pretrain-encoder")
@click.option(
    "--prepared",
    "prepared_dir",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=REPO_ROOT / "prepared",
    show_default=True,
    help="Directory produced by `lorcana-train prepare`.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=REPO_ROOT / "artifacts" / "encoder",
    show_default=True,
    help="Where to write the trained encoder + tokeniser + run logs.",
)
@click.option("--epochs", type=int, default=40, show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--learning-rate", type=float, default=3e-4, show_default=True)
@click.option("--patience", type=int, default=5, show_default=True)
@click.option("--device", type=str, default=None, help="cuda / mps / cpu; auto-detected when unset.")
@click.option("--seed", type=int, default=0, show_default=True)
def pretrain_encoder_cmd(
    prepared_dir: Path,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device: str | None,
    seed: int,
) -> None:
    """Pretrain the card encoder (MLM on text + denoising AE on struct)."""
    opts = PretrainOptions(
        prepared_dir=prepared_dir,
        out_dir=out_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience,
        device=device,
        seed=seed,
    )
    result = run_pretrain(opts)
    click.echo(
        f"pretrain-encoder: best epoch {result.best_epoch}, "
        f"held-out total {result.best_heldout_total:.4f}. wrote {result.out_dir}"
    )


@main.command("export-encoder")
@click.option(
    "--checkpoint",
    "checkpoint_dir",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=REPO_ROOT / "artifacts" / "encoder",
    show_default=True,
    help="Directory produced by `lorcana-train pretrain-encoder`.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=REPO_ROOT / "artifacts" / "encoder-export",
    show_default=True,
    help="Where to write the export bundle (encoder-manifest.json + artifacts).",
)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--device", type=str, default=None, help="cuda / mps / cpu; auto-detected when unset.")
def export_encoder_cmd(
    checkpoint_dir: Path,
    out_dir: Path,
    batch_size: int,
    device: str | None,
) -> None:
    """Export card embeddings + weights + manifest from a trained checkpoint."""
    opts = ExportOptions(
        checkpoint_dir=checkpoint_dir,
        out_dir=out_dir,
        batch_size=batch_size,
        device=device,
    )
    result = run_export(opts)
    click.echo(
        f"export-encoder: {result.card_count} cards -> "
        f"{result.embedding_shape[0]}x{result.embedding_shape[1]} embeddings. "
        f"wrote {result.manifest_path}"
    )


@main.command()
def train() -> None:
    """Train the proposal net and per-step evaluator."""
    raise NotImplementedError


@main.command()
def evaluate() -> None:
    """Run the quality-gate evaluation suite."""
    raise NotImplementedError


@main.command()
def export() -> None:
    """Export trained models to ONNX + write the manifest."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
