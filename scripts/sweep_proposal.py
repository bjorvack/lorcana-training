#!/usr/bin/env python3
"""Tiny hyperparameter sweep for the proposal net.

Runs a fixed list of configurations back-to-back, captures the per-run
``ProposalResult`` + last-epoch history, and writes a
``artifacts/proposal-sweep/summary.json`` with the best held-out
metric for each config. No fancy search — we're looking for a signal
on which lever matters (shrink, per-mask target, wider split) not a
full grid. Total runtime on CPU ≈ 5 runs × ~2 min ≈ 10 min.

Usage:

    uv run python scripts/sweep_proposal.py [--device cpu|mps|cuda]

Each config writes its own ``artifacts/proposal-sweep/<name>/`` dir
with the usual checkpoint + run log + manifest, so you can cherry-pick
the winner into ``artifacts/proposal/`` without re-training.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

from lorcana_training.config import REPO_ROOT
from lorcana_training.proposal import ProposalOptions, TargetMode, train_proposal


SWEEP_ROOT = REPO_ROOT / "artifacts" / "proposal-sweep"


def _baseline_opts() -> ProposalOptions:
    # DESIGN.md defaults. Same as the first training run so the sweep
    # has a reference point on this exact machine / dataset.
    return ProposalOptions(
        out_dir=SWEEP_ROOT / "baseline",
        epochs=30,
        patience=6,
        device=None,
        seed=0,
    )


def _shrink_regularise(base: ProposalOptions) -> ProposalOptions:
    return ProposalOptions(
        **{
            **asdict(base),
            "out_dir": SWEEP_ROOT / "shrink-reg",
            "d_model": 192,
            "n_layers": 4,
            "ff_dim": 768,
            "dropout": 0.2,
            "weight_decay": 0.05,
        }
    )


def _one_hot_target(base: ProposalOptions) -> ProposalOptions:
    return ProposalOptions(
        **{
            **asdict(base),
            "out_dir": SWEEP_ROOT / "one-hot",
            "target_mode": TargetMode.ONE_HOT_REMOVED,
        }
    )


def _wider_split(base: ProposalOptions) -> ProposalOptions:
    return ProposalOptions(
        **{
            **asdict(base),
            "out_dir": SWEEP_ROOT / "wider-split",
            "train_split": "train.evaluator.jsonl",
        }
    )


def _all_three(base: ProposalOptions) -> ProposalOptions:
    return ProposalOptions(
        **{
            **asdict(base),
            "out_dir": SWEEP_ROOT / "all-three",
            "d_model": 192,
            "n_layers": 4,
            "ff_dim": 768,
            "dropout": 0.2,
            "weight_decay": 0.05,
            "target_mode": TargetMode.ONE_HOT_REMOVED,
            "train_split": "train.evaluator.jsonl",
        }
    )


def _run(name: str, opts: ProposalOptions) -> dict[str, object]:
    print(f"\n=== sweep: {name} ===")
    print(f"    train_split={opts.train_split} target_mode={opts.target_mode.value}")
    print(
        f"    d_model={opts.d_model} n_layers={opts.n_layers} "
        f"dropout={opts.dropout} weight_decay={opts.weight_decay}"
    )
    start = time.monotonic()
    result = train_proposal(opts)
    elapsed = time.monotonic() - start
    # Pull the last history entry for extra context (CE curve endpoints).
    run_path = opts.out_dir / "proposal-run.json"
    history = json.loads(run_path.read_text(encoding="utf8"))["history"]
    return {
        "name": name,
        "elapsed_s": elapsed,
        "best_epoch": result.best_epoch,
        "best_heldout_total": result.best_heldout_total,
        "best_heldout_ce": result.best_heldout_ce,
        "best_heldout_entropy": result.best_heldout_entropy,
        "grad_param_count": result.gradient_parameter_count,
        "n_epochs_run": len(history),
        "train_ce_final": history[-1].get("train_ce") if history else None,
        "heldout_ce_final": history[-1].get("heldout_ce") if history else None,
        "out_dir": str(opts.out_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None, help="cpu / mps / cuda")
    parser.add_argument(
        "--only",
        type=str,
        nargs="*",
        help="Run only these named configs (default: all).",
    )
    args = parser.parse_args()

    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

    baseline = _baseline_opts()
    if args.device:
        baseline = ProposalOptions(**{**asdict(baseline), "device": args.device})

    configs: dict[str, ProposalOptions] = {
        "baseline": baseline,
        "shrink-reg": _shrink_regularise(baseline),
        "one-hot": _one_hot_target(baseline),
        "wider-split": _wider_split(baseline),
        "all-three": _all_three(baseline),
    }

    picked = args.only or list(configs.keys())
    missing = [n for n in picked if n not in configs]
    if missing:
        raise SystemExit(f"Unknown config(s): {missing}")

    results: list[dict[str, object]] = []
    for name in picked:
        results.append(_run(name, configs[name]))
        # Persist after each run so a crash doesn't cost completed work.
        summary_path = SWEEP_ROOT / "summary.json"
        summary_path.write_text(json.dumps({"runs": results}, indent=2) + "\n", encoding="utf8")

    print("\n=== sweep summary (sorted by best_heldout_total) ===")
    for r in sorted(results, key=lambda x: x["best_heldout_total"]):  # type: ignore[arg-type]
        print(
            f"  {r['name']:12s}  "
            f"total={r['best_heldout_total']:.4f}  "
            f"CE={r['best_heldout_ce']:.4f}  "
            f"H={r['best_heldout_entropy']:.2f}  "
            f"best_epoch={r['best_epoch']}  "
            f"t={r['elapsed_s']:.0f}s"
        )


if __name__ == "__main__":
    main()
