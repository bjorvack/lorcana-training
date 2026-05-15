# AGENTS.md — lorcana-training

## Pre-commit checklist (run before every commit/push)

CI runs these exact commands. Skipping them locally means
shipping red commits to `main`. **Always run both before
`git commit`:**

```bash
uv run ruff check .
uv run pytest
```

If ruff reports auto-fixable issues:

```bash
uv run ruff check --fix .
uv run ruff format .
```

Then re-run `uv run ruff check .` to confirm.

## Schemas codegen

When `schemas_release_tag` is bumped (or schema types change
upstream), regenerate the pydantic models and commit the diff:

```bash
uv run python -m lorcana_training.schemas.gen
git diff src/lorcana_training/schemas/generated   # should be clean after re-run
```

CI fails if generated models drift from the pinned tag.

## Why this matters

- `.github/workflows/ci.yml` runs `uv run ruff check .` then
  `uv run pytest`, then a "verify generated pydantic models are
  in sync" step. Any failing fails the build.
- A red CI on `main` is a noisy distraction and frequently
  hides genuine downstream regressions on later commits.
