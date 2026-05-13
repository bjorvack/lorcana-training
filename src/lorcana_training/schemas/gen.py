"""Regenerate ``src/lorcana_training/schemas/generated/`` from the
pinned ``lorcana-schemas`` release.

Usage:

    uv run python -m lorcana_training.schemas.gen

What it does:

1. Resolve ``schemas_repo`` + ``schemas_release_tag`` from
   ``config/training.yaml``.
2. Download the source tarball for that tag (cached under
   ``.cache/artifacts/``).
3. Extract ``schemas/*.schema.json`` into a tmp dir.
4. Run ``datamodel-code-generator`` on each schema, writing
   ``src/lorcana_training/schemas/generated/<name>.py``.
5. Write a ``.generated_from`` stamp so humans (and the tests) can
   tell at a glance which schemas tag is currently checked in.

The generated code is committed. CI runs this script in a separate
check to guarantee the checked-in models and the pinned tag stay in
sync; a drift fails the build.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from ..config import REPO_ROOT, load_config
from ..release.download import download_source_tarball


GENERATED_DIR = REPO_ROOT / "src" / "lorcana_training" / "schemas" / "generated"
_HEADER = "# AUTO-GENERATED — do not edit by hand. See lorcana_training.schemas.gen.\n"
# `card-set.schema.json` → module name `card_set` (python-friendly).
_SCHEMA_TO_MODULE: dict[str, str] = {
    "card": "card",
    "card-set": "card_set",
    "deck": "deck",
    "tournament": "tournament",
    "dataset": "dataset",
    "manifest": "manifest",
}


def _extract_schemas(tarball: Path, dest: Path) -> dict[str, Path]:
    """Return {schema_name: path/to/<name>.schema.json}."""
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(dest, filter="data")  # `data` strips symlinks / abs paths
    out: dict[str, Path] = {}
    for p in (dest / "schemas").glob("*.schema.json"):
        out[p.name.removesuffix(".schema.json")] = p
    return out


def _run_codegen(schema_path: Path, output: Path, class_name: str) -> None:
    """Invoke ``datamodel-codegen`` as a subprocess.

    Using the CLI (rather than its Python API) keeps us insulated from
    ``datamodel-code-generator``'s internals — the CLI surface is the
    only thing they treat as stable.
    """
    # `--collapse-root-models` turns top-level `$ref` roots (how zod's
    # JSON schema export wraps every top-level schema) into a single
    # pydantic model rather than a pointless alias + wrapper.
    cmd = [
        sys.executable,
        "-m",
        "datamodel_code_generator",
        "--input",
        str(schema_path),
        "--input-file-type",
        "jsonschema",
        "--output",
        str(output),
        "--output-model-type",
        "pydantic_v2.BaseModel",
        "--target-python-version",
        "3.12",
        "--collapse-root-models",
        "--use-schema-description",
        "--use-title-as-name",
        "--disable-timestamp",
        "--snake-case-field",
        "--class-name",
        class_name,
        "--strict-nullable",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"datamodel-codegen failed for {schema_path}:\n{result.stdout}\n{result.stderr}"
        )


def _prepend_header(path: Path, tag: str) -> None:
    original = path.read_text(encoding="utf8")
    path.write_text(
        f"{_HEADER}# Generated from lorcana-schemas@{tag}.\n\n{original}", encoding="utf8"
    )


def regenerate() -> None:
    cfg = load_config()
    repo, tag = cfg.schemas_repo, cfg.schemas_release_tag
    tarball = download_source_tarball(repo, tag)

    # Clean the target directory before re-generating so stale modules
    # from a previous run don't silently linger.
    if GENERATED_DIR.exists():
        shutil.rmtree(GENERATED_DIR)
    GENERATED_DIR.mkdir(parents=True)
    (GENERATED_DIR / "__init__.py").write_text(
        f'{_HEADER}"""Pydantic models generated from lorcana-schemas@{tag}."""\n',
        encoding="utf8",
    )

    with tempfile.TemporaryDirectory() as tmp:
        extracted = _extract_schemas(tarball, Path(tmp))
        missing = set(_SCHEMA_TO_MODULE) - set(extracted)
        if missing:
            raise FileNotFoundError(f"tarball is missing schemas: {sorted(missing)}")

        for schema_name, module_name in _SCHEMA_TO_MODULE.items():
            output = GENERATED_DIR / f"{module_name}.py"
            class_name = "".join(part.capitalize() for part in schema_name.split("-"))
            _run_codegen(extracted[schema_name], output, class_name)
            _prepend_header(output, tag)
            print(f"[schemas.gen] wrote {output.relative_to(REPO_ROOT)}")

    (GENERATED_DIR / ".generated_from").write_text(f"{repo}@{tag}\n", encoding="utf8")

    # Normalise quote style + trailing commas through `ruff format`.
    # Without this the drift check flakes across datamodel-codegen
    # patch versions, which silently flip between single and double
    # quotes depending on upstream's chosen `black` version of the
    # week. ruff format is deterministic, so committing its output
    # makes the drift check a real content diff rather than a
    # cosmetic one.
    format_result = subprocess.run(
        ["ruff", "format", str(GENERATED_DIR)],
        capture_output=True,
        text=True,
    )
    if format_result.returncode != 0:
        raise RuntimeError(
            f"ruff format failed on generated dir:\n{format_result.stdout}\n{format_result.stderr}",
        )

    print(f"[schemas.gen] ok, pinned to {repo}@{tag}")


if __name__ == "__main__":
    regenerate()
