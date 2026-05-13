"""Tests for the prepare orchestrator.

The full prepare() run is covered by the opt-in network test below.
Non-network tests stub out the download step to keep the suite fast.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from lorcana_training.config import load_config
from lorcana_training.prepare import PrepareOptions, prepare
from lorcana_training.schemas.generated.card_set import CardSet
from lorcana_training.schemas.generated.dataset import Dataset


def _card(id_: str, name: str, **overrides: object) -> dict:
    base = {
        "id": id_,
        "name": name,
        "version": None,
        "setCode": "1",
        "cardNumber": 1,
        "cost": 3,
        "inkwell": True,
        "inks": ["Amber"],
        "types": ["Character"],
        "classifications": ["Storyborn"],
        "keywords": [],
        "text": "",
        "flavor": None,
        "imageUrl": f"https://example.test/{id_}.avif",
        "legality": "legal",
        "lore": 2,
        "strength": 3,
        "willpower": 3,
        "moveCost": None,
    }
    base.update(overrides)
    return base


def _fake_inputs() -> tuple[CardSet, Dataset]:
    cards = [_card(f"crd_{chr(ord('a') + i)}", f"Card{i}", cardNumber=i + 1) for i in range(15)]
    cs = CardSet.model_validate(
        {
            "cardSetVersion": "sha256:fake",
            "fetchedAt": "2026-05-13T00:00:00Z",
            "cards": cards,
        }
    )
    decks = [
        {
            "placement": 1,
            "player": "p1",
            "deck": {
                "inks": ["Amber"],
                "cards": [{"cardId": c["id"], "count": 4} for c in cards],
                "name": None,
                "source": "t",
            },
        }
    ]
    ds = Dataset.model_validate(
        {
            "datasetVersion": "0.1.0",
            "schemaVersion": "0.4.0",
            "cardSetVersion": "sha256:fake",
            "cardsReleaseTag": "cards-vfake",
            "generatedAt": "2026-05-13T00:00:00Z",
            "sources": ["t"],
            "tournaments": [
                {
                    "sourceUrl": "https://example.test/t",
                    "sourceName": "t",
                    "name": "T",
                    "date": "2025-06-01",
                    "decks": decks,
                }
            ],
        }
    )
    return cs, ds


def test_prepare_stubbed_end_to_end(tmp_path: Path) -> None:
    cs, ds = _fake_inputs()
    with (
        patch(
            "lorcana_training.prepare.run.download_cards",
            return_value=(tmp_path / "cards.json", cs),
        ),
        patch(
            "lorcana_training.prepare.run.download_tournaments",
            return_value=(tmp_path / "dataset.json", ds),
        ),
    ):
        result = prepare(PrepareOptions(out_dir=tmp_path / "prepared"))
    assert not result.cached
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text())
    # Sources carry the exact pins from config/training.yaml for traceability.
    cfg = load_config()
    assert manifest["sources"]["cardsReleaseTag"] == cfg.cards_release_tag
    assert manifest["sources"]["tournamentsReleaseTag"] == cfg.tournaments_release_tag
    # Outputs cover every expected artifact.
    assert set(manifest["outputs"]) == {
        "vocab",
        "printing_to_logical",
        "dedup_report",
        "card_features",
        "feature_schema",
        "train_proposal",
        "train_evaluator",
        "heldout",
        "splits_report",
    }


def test_prepare_cache_hit_is_a_noop(tmp_path: Path) -> None:
    cs, ds = _fake_inputs()
    prep_dir = tmp_path / "prepared"
    with (
        patch(
            "lorcana_training.prepare.run.download_cards",
            return_value=(tmp_path / "cards.json", cs),
        ) as dc,
        patch(
            "lorcana_training.prepare.run.download_tournaments",
            return_value=(tmp_path / "dataset.json", ds),
        ) as dt,
    ):
        first = prepare(PrepareOptions(out_dir=prep_dir))
        assert not first.cached
        assert dc.call_count == 1 and dt.call_count == 1

        second = prepare(PrepareOptions(out_dir=prep_dir))
        assert second.cached
        assert second.content_hash == first.content_hash
        # Downloaders untouched on the cache hit.
        assert dc.call_count == 1 and dt.call_count == 1

        third = prepare(PrepareOptions(out_dir=prep_dir, force=True))
        assert not third.cached
        assert dc.call_count == 2 and dt.call_count == 2


@pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1", reason="set RUN_NETWORK_TESTS=1 to enable"
)
def test_prepare_live_pinned_artifacts(tmp_path: Path) -> None:
    result = prepare(PrepareOptions(out_dir=tmp_path / "prepared"))
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text())
    # Pinned tournaments-v0.3.0 has 6137 decks.
    assert manifest["validation"]["totalDecks"] == 6137
