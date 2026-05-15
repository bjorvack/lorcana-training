# lorcana-training

Trains the **card encoder**, **proposal net**, and **per-step evaluator**
that power the on-device deck-building assistant in
[`lorcana-web`](https://github.com/bjorvack/lorcana-web), then exports
them to ONNX so they run client-side in a browser Web Worker.

Inputs are the public artifacts produced by
[`lorcana-scraper`](https://github.com/bjorvack/lorcana-scraper):

- `cards-vN` — full Lorcana card pool (text + structured features).
- `tournaments-vN` — real tournament decks deck-resolved against
  `cards-vN`.

Output is a `model-vN` GitHub Release containing a self-contained
ONNX bundle (encoder + proposal + evaluator + tokenizer + tables).
See [`DESIGN.md`](./DESIGN.md) for the full architecture.

## Pipeline

```
prepare         normalise inputs → train/val splits + tokenizer
   │
   ▼
pretrain-encoder    BERT-style masked-card pretraining on (card-text → card)
   │
   ▼
export-encoder      freeze + save card embedding table for downstream nets
   │
   ▼
train-proposal      next-card prediction conditioned on partial deck (top-k)
   │
   ▼
train-evaluator     scores partial-deck states (used to rank proposals)
   │
   ▼
build-tables        precompute per-card lookup tables to avoid runtime tokenization
   │
   ▼
export              ONNX-export all three nets + bundle artifacts
```

Each step is a `lorcana-train <subcommand>` and a deterministic step in
the [`train.yml`](.github/workflows/train.yml) GitHub Actions workflow.

## Quick start

```bash
uv sync
# Pull the pinned cards-vN + tournaments-vN into ./data
uv run lorcana-train prepare
uv run lorcana-train pretrain-encoder
uv run lorcana-train export-encoder
uv run lorcana-train train-proposal
uv run lorcana-train train-evaluator
uv run lorcana-train build-tables
uv run lorcana-train export
```

End-to-end on a single CPU runner: ~1-2 h with default hyperparams.

## CLI

```bash
uv run lorcana-train --help
```

| Subcommand | Reads | Writes |
|---|---|---|
| `prepare` | `cards-vN`, `tournaments-vN` | `data/{train,val}.jsonl`, tokenizer |
| `pretrain-encoder` | training data + tokenizer | encoder checkpoint |
| `export-encoder` | encoder checkpoint | per-card embedding table |
| `train-proposal` | tournament decks + embeddings | proposal checkpoint |
| `train-evaluator` | tournament decks + embeddings | evaluator checkpoint |
| `build-tables` | embeddings + nets | inference-time lookup tables |
| `export` | all checkpoints + tables | ONNX bundle ready for `lorcana-web` |

All subcommands accept `--config-path` (default `config/training.yaml`)
and write into `runs/<run-id>/` with deterministic seeds.

## Schemas

`@bjorvack/lorcana-schemas` is the source of truth for the input
shapes. We pin a release tag in `pyproject.toml` and run
`datamodel-code-generator` to produce typed `pydantic` models in
`src/lorcana_training/schemas/generated/`. CI fails the build if the
generated models drift from the pinned tag, so a schemas bump is a
two-step PR:

```bash
# 1. Bump pyproject.toml schemas_release_tag = "vX.Y.Z"
# 2. Re-run codegen and commit the diff
uv run python -m lorcana_training.schemas.gen
git diff src/lorcana_training/schemas/generated   # should be clean
```

## Develop

See [`AGENTS.md`](./AGENTS.md) for the mandatory pre-commit checklist.

```bash
uv sync                         # install incl. dev extras
uv run ruff check .             # lint
uv run pytest                   # tests (currently 127 + 5 skipped)
```

## CI

| Workflow | Trigger | Output |
|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | every push / PR | ruff + pytest + schemas-codegen drift check + live smoke (on `main` / labelled PRs) |
| [`pretrain.yml`](.github/workflows/pretrain.yml) | manual dispatch | encoder checkpoint as a workflow artifact |
| [`train.yml`](.github/workflows/train.yml) | weekly cron + dispatch + push to `main` (config / pipeline) | full retrain end-to-end |
| [`release.yml`](.github/workflows/release.yml) | tag / workflow output | publish `model-vN` GitHub Release |
| [`new-set-reminder.yml`](.github/workflows/new-set-reminder.yml) | weekly cron | auto-merging PR when a fresh `cards-vN` lands upstream |

The `train.yml` workflow runs `prepare → … → export` on a CPU runner
and publishes a non-prerelease `model-v<auto-version>` GitHub Release
that `lorcana-web`'s daily reminder picks up on its next pass.
