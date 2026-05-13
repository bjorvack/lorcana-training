"""Shape + behaviour tests for the CardEncoder torch module."""

from __future__ import annotations

import pytest
import torch

from lorcana_training.models.card_encoder import (
    CardEncoder,
    CardEncoderConfig,
    StructuredEncoder,
    TextEncoder,
)


def _cfg(**overrides: object) -> CardEncoderConfig:
    base = {
        "vocab_size": 256,
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 2,
        "ff_dim": 32,
        "max_positions": 32,
        "text_dim": 24,
        "struct_feature_dim": 40,
        "struct_hidden": 32,
        "struct_dim": 12,
        "encoder_dim": 48,
    }
    base.update(overrides)
    return CardEncoderConfig(**base)  # type: ignore[arg-type]


def test_structured_encoder_shape() -> None:
    cfg = _cfg()
    m = StructuredEncoder(cfg)
    out = m(torch.randn(3, cfg.struct_feature_dim))
    assert out.shape == (3, cfg.struct_dim)


def test_text_encoder_masks_padding() -> None:
    cfg = _cfg()
    m = TextEncoder(cfg)
    # Batch of 2 rows; row 0 has two tokens then pad, row 1 all pad.
    token_ids = torch.tensor([[10, 11, cfg.pad_token_id], [cfg.pad_token_id] * 3])
    out = m(token_ids)
    assert out.shape == (2, cfg.text_dim)
    # The all-pad row should not NaN or blow up; it's a graceful zero-path.
    assert torch.isfinite(out).all()


def test_card_encoder_end_to_end_shape() -> None:
    cfg = _cfg()
    enc = CardEncoder(cfg)
    token_ids = torch.randint(1, cfg.vocab_size, (5, cfg.max_positions // 2))
    struct = torch.randn(5, cfg.struct_feature_dim)
    out = enc(token_ids, struct)
    assert out.shape == (5, cfg.encoder_dim)
    # L2-normalised by default.
    norms = out.pow(2).sum(dim=-1).sqrt()
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_unnormalised_output_is_raw_fusion() -> None:
    cfg = _cfg()
    enc = CardEncoder(cfg)
    token_ids = torch.randint(1, cfg.vocab_size, (2, 8))
    struct = torch.randn(2, cfg.struct_feature_dim)
    out = enc(token_ids, struct, normalise=False)
    # Not unit-norm in general.
    assert not torch.allclose(out.pow(2).sum(-1).sqrt(), torch.ones(2), atol=1e-3)


def test_backward_pass_updates_every_parameter() -> None:
    cfg = _cfg()
    enc = CardEncoder(cfg)
    token_ids = torch.randint(1, cfg.vocab_size, (4, 16))
    struct = torch.randn(4, cfg.struct_feature_dim)
    out = enc(token_ids, struct)
    loss = out.pow(2).sum()
    loss.backward()
    # Every parameter either has a gradient or was frozen; our module
    # has no frozen params, so all must have non-None grads.
    missing = [n for n, p in enc.named_parameters() if p.grad is None]
    assert missing == []


def test_seq_len_over_max_positions_errors() -> None:
    cfg = _cfg(max_positions=8)
    enc = CardEncoder(cfg)
    too_long = torch.randint(1, cfg.vocab_size, (1, cfg.max_positions + 1))
    with pytest.raises(ValueError):
        enc(too_long, torch.randn(1, cfg.struct_feature_dim))


def test_parameter_count_reasonable_for_design_defaults() -> None:
    """Ballpark check on the DESIGN.md sizing. Not a tight bound."""
    cfg = CardEncoderConfig(
        vocab_size=4_200,  # current real pool
        struct_feature_dim=60,  # current real schema dim
    )
    enc = CardEncoder(cfg)
    params = enc.parameter_count
    # Should be in the low millions, well under the DESIGN target of ~5M.
    assert 500_000 < params < 5_000_000, params
