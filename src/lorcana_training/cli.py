"""Command-line entry point: ``lorcana-train``.

TODO: wire subcommands (pretrain, train, evaluate, export, release).
"""

from __future__ import annotations

import click


@click.group()
def main() -> None:
    """Lorcana training CLI."""


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
