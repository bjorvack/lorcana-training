"""Unit tests for the play-frequency + archetype-centroid tables.

Covers each building block + an end-to-end build_tables smoke run.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.numpy import save_file as save_numpy_file

from lorcana_training.proposal.data import Deck
from lorcana_training.tables import BuildTablesOptions, build_tables
from lorcana_training.tables.archetype_centroids import (
    compute_deck_vectors,
    kmeans,
    write_archetype_centroids_json,
)
from lorcana_training.tables.play_frequency import (
    ALL_KEY,
    compute_play_frequency,
    write_play_frequency_json,
)


def _deck(cards: list[tuple[int, int]], inks: list[str]) -> Deck:
    return Deck(cards=tuple(sorted(cards)), inks=tuple(inks))


def test_play_frequency_respects_counts_and_ink_key() -> None:
    decks = [
        _deck([(1, 4), (2, 4)], ["amber", "ruby"]),
        _deck([(1, 2), (3, 4)], ["amber", "ruby"]),
        _deck([(4, 4), (5, 4)], ["emerald", "steel"]),
    ]
    table = compute_play_frequency(decks)
    # Keys are canonical sorted ink-pair strings.
    assert "amber|ruby" in table
    assert "emerald|steel" in table
    assert ALL_KEY in table
    # Fractions sum to 1 per ink-pair (up to float slop).
    for ink_key, row in table.items():
        assert abs(sum(row.values()) - 1.0) < 1e-9, ink_key
    # Amber/ruby totals: card 1 appears 6 times, card 2 appears 4, card 3 appears 4.
    amber_ruby = table["amber|ruby"]
    assert amber_ruby[1] == 6 / 14
    assert amber_ruby[2] == 4 / 14
    assert amber_ruby[3] == 4 / 14


def test_play_frequency_written_as_stable_json(tmp_path: Path) -> None:
    decks = [
        _deck([(1, 4), (2, 4)], ["ruby", "amber"]),  # unsorted inks → still "amber|ruby"
        _deck([(1, 2), (3, 4)], ["amber", "ruby"]),
    ]
    table = compute_play_frequency(decks)
    out = tmp_path / "play_frequency.json"
    write_play_frequency_json(table, out)
    # Byte-identical on re-run.
    out_b = tmp_path / "again.json"
    write_play_frequency_json(table, out_b)
    assert out.read_bytes() == out_b.read_bytes()
    payload = json.loads(out.read_text())
    # Keys are strings (card ids are stringified for JSON).
    first_row = payload["amber|ruby"]
    assert set(first_row.keys()) <= {"1", "2", "3"}


def test_compute_deck_vectors_l2_normalised() -> None:
    decks = [
        _deck([(1, 4), (2, 4)], ["amber", "ruby"]),
        _deck([(3, 4)], ["amber", "ruby"]),
    ]
    torch.manual_seed(0)
    embeddings = torch.randn(6, 8)
    vecs = compute_deck_vectors(decks, embeddings)
    assert vecs.shape == (2, 8)
    # Each row should have unit norm.
    norms = vecs.norm(dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_kmeans_returns_k_centroids_and_assigns_all() -> None:
    torch.manual_seed(0)
    # Make three obvious clusters in 4-D by offsetting a random
    # base and adding tiny noise.
    centres = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
        ],
    )
    points = []
    for c in centres:
        points.append(c.expand(30, -1) + 0.1 * torch.randn(30, 4))
    data = torch.cat(points, dim=0)
    data = data / data.norm(dim=1, keepdim=True)
    result = kmeans(data, k=3, seed=1)
    assert result.centroids.shape == (3, 4)
    # Centroids should be unit-norm (we normalise on exit).
    norms = result.centroids.norm(dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-4)
    assert sum(result.cluster_sizes) == data.shape[0]
    assert all(size > 0 for size in result.cluster_sizes)


def test_build_tables_end_to_end(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    vocab_size = 6
    (prepared / "vocab.json").write_text(
        json.dumps(
            {
                "padIndex": 0,
                "size": vocab_size,
                "cardSetVersion": "sha256:test",
                "cardsReleaseTag": "cards-vTEST",
            }
        ),
        encoding="utf8",
    )
    (prepared / "manifest.json").write_text(
        json.dumps(
            {
                "contentHash": "sha256:abc",
                "sources": {
                    "cardsReleaseTag": "cards-vTEST",
                    "cardSetVersion": "sha256:test",
                },
            }
        ),
        encoding="utf8",
    )
    # 4 decks across two ink pairs.
    with (prepared / "train.evaluator.jsonl").open("w", encoding="utf8") as f:
        for row in [
            {"cards": [[1, 4], [2, 4]], "inks": ["amber", "ruby"]},
            {"cards": [[1, 4], [3, 4]], "inks": ["amber", "ruby"]},
            {"cards": [[4, 4], [5, 4]], "inks": ["emerald", "steel"]},
            {"cards": [[4, 2], [6, 4]], "inks": ["emerald", "steel"]},
        ]:
            f.write(json.dumps(row) + "\n")

    encoder = tmp_path / "encoder-export"
    encoder.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(vocab_size + 1, 8, generator=g).numpy().astype("float32")
    save_numpy_file({"card_embeddings": emb}, str(encoder / "card_embeddings.fp32.safetensors"))
    (encoder / "encoder-manifest.json").write_text(
        json.dumps({"sources": {"cardsReleaseTag": "cards-vTEST"}}),
        encoding="utf8",
    )

    opts = BuildTablesOptions(
        prepared_dir=prepared,
        encoder_export_dir=encoder,
        out_dir=tmp_path / "tables",
        k_archetypes=2,  # tiny dataset → tiny k
        seed=7,
    )
    result = build_tables(opts)

    assert result.play_frequency_path.exists()
    assert result.archetype_centroids_path.exists()
    assert result.manifest_path.exists()
    assert result.n_decks == 4
    assert result.k_effective == 2

    freq = json.loads(result.play_frequency_path.read_text())
    assert "amber|ruby" in freq
    assert "emerald|steel" in freq
    assert "_all" in freq

    arch = json.loads(result.archetype_centroids_path.read_text())
    assert arch["k"] == 2
    assert arch["dim"] == 8
    assert len(arch["centroids"]) == 2

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["vocabSize"] == vocab_size
    assert manifest["playFrequency"]["sha256"].startswith("sha256:")
    assert manifest["archetypeCentroids"]["sha256"].startswith("sha256:")
    # write helper consumed without producing artifacts that shouldn't
    # linger in tmp_path.
    _ = write_archetype_centroids_json
