"""Tests for the pretrain heads + masking utilities."""

from __future__ import annotations

import torch

from lorcana_training.cards.features import build_feature_schema, build_features
from lorcana_training.cards.logical import build_logical_cards
from lorcana_training.cards.vocab import build_vocab
from lorcana_training.models.card_encoder import CardEncoder, CardEncoderConfig
from lorcana_training.models.pretrain_heads import (
    MLM_IGNORE_INDEX,
    MlmHead,
    StructReconstructionHead,
    compute_pretrain_loss,
)
from lorcana_training.schemas.generated.card_set import CardSet
from lorcana_training.train.masking import mask_structured_blocks, mask_tokens


def _small_cfg(struct_dim: int = 40, vocab: int = 64) -> CardEncoderConfig:
    return CardEncoderConfig(
        vocab_size=vocab,
        d_model=16,
        n_heads=4,
        n_layers=2,
        ff_dim=32,
        max_positions=32,
        text_dim=24,
        struct_feature_dim=struct_dim,
        struct_hidden=32,
        struct_dim=12,
        encoder_dim=48,
    )


def test_mask_tokens_excludes_special_and_respects_prob() -> None:
    torch.manual_seed(0)
    token_ids = torch.tensor([[0, 1, 2, 3, 4, 10, 11, 12, 13, 14]])
    specials = {0, 1, 2, 3, 4}  # PAD/UNK/CLS/SEP/MASK
    result = mask_tokens(
        token_ids,
        mask_token_id=4,
        vocab_size=64,
        special_token_ids=specials,
        mask_prob=1.0,  # mask every eligible position for a deterministic check
    )
    # Labels hold the original for masked positions, IGNORE for specials.
    assert (result.labels[0, :5] == MLM_IGNORE_INDEX).all()
    assert (result.labels[0, 5:] == torch.tensor([10, 11, 12, 13, 14])).all()
    # No special token has been corrupted.
    assert (result.input_ids[0, :5] == token_ids[0, :5]).all()


def test_mask_tokens_deterministic_with_seed() -> None:
    token_ids = torch.randint(10, 50, (2, 16))
    specials = {0, 1, 2, 3, 4}
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = mask_tokens(
        token_ids, mask_token_id=4, vocab_size=64, special_token_ids=specials, generator=g1
    )
    b = mask_tokens(
        token_ids, mask_token_id=4, vocab_size=64, special_token_ids=specials, generator=g2
    )
    assert torch.equal(a.input_ids, b.input_ids)
    assert torch.equal(a.labels, b.labels)


def _fake_schema():
    # Build a small real schema off one-card fixtures so the tests
    # exercise the actual FeatureSchema layout rather than a mock.
    cards = [
        {
            "id": "crd_a",
            "name": "A",
            "version": None,
            "setCode": "1",
            "cardNumber": 1,
            "cost": 3,
            "inkwell": True,
            "inks": ["Amber"],
            "types": ["Character"],
            "classifications": ["Hero", "Storyborn"],
            "keywords": ["Shift", "Rush"],
            "text": "",
            "flavor": None,
            "imageUrl": "https://example.test/a.avif",
            "legality": "legal",
            "lore": 2,
            "strength": 3,
            "willpower": 3,
            "moveCost": None,
        }
    ]
    cs = CardSet.model_validate(
        {"cardSetVersion": "sha256:test", "fetchedAt": "2026-05-13T00:00:00Z", "cards": cards}
    )
    logical = build_logical_cards(cs)
    vocab = build_vocab(logical)
    schema = build_feature_schema(logical.cards)
    feats = build_features(vocab, schema)
    return schema, torch.from_numpy(feats[1:])  # drop PAD row


def test_mask_structured_blocks_zeroes_selected_slices() -> None:
    schema, feats = _fake_schema()
    g = torch.Generator().manual_seed(42)
    # Drop everything so the result is all zeros, proves the indices are right.
    result = mask_structured_blocks(feats, schema=schema, block_drop_prob=1.0, generator=g)
    assert result.features.shape == feats.shape
    assert torch.allclose(result.features, torch.zeros_like(feats))


def test_mask_structured_blocks_drop_prob_zero_is_identity() -> None:
    schema, feats = _fake_schema()
    result = mask_structured_blocks(feats, schema=schema, block_drop_prob=0.0)
    assert torch.equal(result.features, feats)
    assert result.dropped_blocks == []


def test_pretrain_loss_end_to_end_backward() -> None:
    cfg = _small_cfg()
    encoder = CardEncoder(cfg)
    mlm_head = MlmHead(encoder)
    struct_head = StructReconstructionHead(cfg)

    token_ids = torch.randint(5, cfg.vocab_size, (4, 16))
    mlm = mask_tokens(
        token_ids,
        mask_token_id=4,
        vocab_size=cfg.vocab_size,
        special_token_ids={0, 1, 2, 3, 4},
    )
    struct_target = torch.rand(4, cfg.struct_feature_dim)
    struct_input = struct_target * (torch.rand_like(struct_target) > 0.3).float()

    losses = compute_pretrain_loss(
        encoder=encoder,
        mlm_head=mlm_head,
        struct_head=struct_head,
        token_ids=mlm.input_ids,
        mlm_labels=mlm.labels,
        struct_features_masked=struct_input,
        struct_features_target=struct_target,
    )
    assert losses.mlm.item() > 0
    assert losses.struct.item() >= 0
    losses.total.backward()
    # Sanity: encoder params and head params all received gradients.
    for m in (encoder, mlm_head, struct_head):
        missing = [n for n, p in m.named_parameters() if p.grad is None]
        assert missing == [], missing


def test_mlm_head_weight_tied_to_token_embedding() -> None:
    cfg = _small_cfg()
    encoder = CardEncoder(cfg)
    mlm_head = MlmHead(encoder)
    # Tied weights: identity on the parameter tensor.
    assert mlm_head.decoder.weight is encoder.text.token_emb.weight
