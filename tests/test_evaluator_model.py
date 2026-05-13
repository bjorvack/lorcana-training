"""Shape + behaviour tests for the Evaluator torch module.

Structurally parallel to ``test_proposal_model.py`` — the two modules
share enough architecture that covering the unique bits (candidate
MLP branch, fusion head, sigmoid logits contract) in their own file
is enough to stay confident when either moves."""

from __future__ import annotations

import pytest
import torch

from lorcana_training.models.evaluator import (
    Evaluator,
    EvaluatorConfig,
    evaluator_loss,
)


def _tiny_cfg(*, vocab_size: int = 16, d_model: int = 16, freeze: bool = True) -> EvaluatorConfig:
    return EvaluatorConfig(
        vocab_size=vocab_size,
        embed_dim=d_model,
        d_model=d_model,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        dropout=0.0,
        freeze_card_embeddings=freeze,
    )


def _random_embeddings(vocab_size: int, dim: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(vocab_size + 1, dim, generator=g)
    emb[0] = 1.0  # non-zero PAD → a broken mask would bias logits
    return emb


def test_evaluator_forward_returns_logits_shape() -> None:
    cfg = _tiny_cfg()
    model = Evaluator(cfg, card_embeddings=_random_embeddings(cfg.vocab_size, cfg.d_model))
    model.eval()
    partial_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]], dtype=torch.long)
    candidate_ids = torch.tensor([7, 9], dtype=torch.long)
    logits = model(partial_ids, candidate_ids)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()


def test_evaluator_padding_invariance() -> None:
    cfg = _tiny_cfg()
    model = Evaluator(cfg, card_embeddings=_random_embeddings(cfg.vocab_size, cfg.d_model))
    model.eval()
    short = torch.tensor([[1, 2, 3]], dtype=torch.long)
    long = torch.tensor([[1, 2, 3, 0, 0, 0]], dtype=torch.long)
    cand = torch.tensor([7], dtype=torch.long)
    with torch.no_grad():
        a = model(short, cand)
        b = model(long, cand)
    assert torch.allclose(a, b, atol=1e-4, rtol=1e-4)


def test_evaluator_frozen_embeddings() -> None:
    cfg = _tiny_cfg(freeze=True)
    model = Evaluator(cfg, card_embeddings=_random_embeddings(cfg.vocab_size, cfg.d_model))
    assert "card_embeddings" in dict(model.named_buffers())
    assert "card_embeddings" not in dict(model.named_parameters())


def test_evaluator_unfrozen_embeddings() -> None:
    cfg = _tiny_cfg(freeze=False)
    model = Evaluator(cfg, card_embeddings=_random_embeddings(cfg.vocab_size, cfg.d_model))
    assert "card_embeddings" in dict(model.named_parameters())
    assert "card_embeddings" not in dict(model.named_buffers())


def test_evaluator_bridges_embed_dim_and_d_model() -> None:
    cfg = EvaluatorConfig(
        vocab_size=8, embed_dim=32, d_model=16, n_heads=4, n_layers=1, ff_dim=32, dropout=0.0
    )
    model = Evaluator(cfg, card_embeddings=torch.randn(cfg.vocab_size + 1, cfg.embed_dim))
    out = model(torch.tensor([[1, 2, 0]], dtype=torch.long), torch.tensor([3], dtype=torch.long))
    assert out.shape == (1,)


def test_evaluator_loss_sign() -> None:
    # High-confidence correct predictions should give loss near 0.
    logits_good = torch.tensor([10.0, -10.0])
    labels = torch.tensor([1.0, 0.0])
    assert evaluator_loss(logits_good, labels).item() < 0.01
    # Inverted predictions should give large loss.
    logits_bad = torch.tensor([-10.0, 10.0])
    assert evaluator_loss(logits_bad, labels).item() > 5.0


def test_evaluator_can_overfit_single_batch() -> None:
    """Trivial overfit: 20 steps should drive BCE near zero on a tiny set."""
    torch.manual_seed(42)
    cfg = _tiny_cfg()
    model = Evaluator(cfg, card_embeddings=_random_embeddings(cfg.vocab_size, cfg.d_model))
    partial_ids = torch.tensor([[1, 2, 3, 4, 5, 0, 0], [6, 7, 8, 9, 10, 11, 0]], dtype=torch.long)
    candidate_ids = torch.tensor([5, 11], dtype=torch.long)
    labels = torch.tensor([1.0, 0.0])

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    initial = evaluator_loss(model(partial_ids, candidate_ids), labels).item()
    for _ in range(40):
        loss = evaluator_loss(model(partial_ids, candidate_ids), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    final = evaluator_loss(model(partial_ids, candidate_ids), labels).item()
    assert final < initial / 3.0, f"BCE should drop 3x, got {initial:.3f} -> {final:.3f}"


def test_evaluator_rejects_mismatched_batch_sizes() -> None:
    cfg = _tiny_cfg()
    model = Evaluator(cfg, card_embeddings=_random_embeddings(cfg.vocab_size, cfg.d_model))
    with pytest.raises(ValueError, match="batch mismatch"):
        model(torch.zeros(3, 4, dtype=torch.long), torch.zeros(2, dtype=torch.long))


def test_evaluator_rejects_bad_head_divisibility() -> None:
    with pytest.raises(ValueError, match="divisible"):
        EvaluatorConfig(vocab_size=4, embed_dim=10, d_model=10, n_heads=3)
