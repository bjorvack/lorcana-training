"""Command-line entry point: ``lorcana-train``."""

from __future__ import annotations

from pathlib import Path

import click

from .config import REPO_ROOT
from .prepare import PrepareOptions, prepare as run_prepare


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


@main.command()
def pretrain() -> None:
    """Pretrain the card encoder."""
    raise NotImplementedError


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
