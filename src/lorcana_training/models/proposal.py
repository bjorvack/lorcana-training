"""Proposal Net — masked set completion over a partial deck.

Architecture (DESIGN.md §2 "Proposal Net"):

    ink_2hot (B, 6)                partial deck (B, N) of card ids (0 = PAD)
         │                                  │
         ▼                                  ▼
     ink MLP (6 → 256)             card_embeddings[ids]  (B, N, 256)
         │                                  │
         └──────── broadcast-add ──── per-position ──────┤
                                                        ▼
                                    ┌────────────────────────────────┐
                                    │ 6-layer Transformer encoder    │
                                    │ d=256, heads=8, no pos-encoding│
                                    │ (it's a set, not a sequence)   │
                                    └──────────────┬─────────────────┘
                                                   ▼
                                     mean+max pool over non-PAD → R^512
                                                   │
                                                   ▼
                                      Linear(512 → |vocab|)  logits

The output is a softmax distribution over card ids — never used directly
at inference, only blended with the evaluator + novelty score inside the
web-side search loop. It is trained to be *informative*, not right:

    L = CE(softmax(logits), target)  −  β · H(softmax(logits))

where the entropy bonus rewards keeping mass on more than one plausible
card (without it the model converges on the meta).

Design choices:

- **Card embeddings are non-trainable by default.** The pre-trained
  encoder is a larger context-aware text+struct network; end-to-end
  fine-tuning during proposal training is possible but not done here
  so the proposal net can be trained against any frozen encoder export.
  The ``freeze_card_embeddings`` flag leaves room for future unfreezing.
- **PAD ids (0) attend to nothing.** We build a key-padding mask so
  zero-padded partials don't pollute mean/max pools.
- **No positional encoding.** Deck is an unordered multiset; adding
  sinusoids or learned positions would inject spurious order signal.
- **Ink conditioning as a broadcast-add, not a prefix token.** Keeps
  the Transformer's input sequence length bounded by the deck size
  and avoids a special case in the padding mask.
- **Weight init matches the pretrain encoder's pre-norm Transformer**
  so stacking the two modules doesn't produce an activation blow-up
  at step zero.

Parameter count for DESIGN defaults (d=256, h=8, layers=6) lands at
≈3.5 M including the 2.3k × 256 embedding table (frozen, not counted
in gradient-bearing params).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Lorcana has exactly 6 inks. The proposal net takes a 6-dim multi-hot
# vector as its deck-level conditioning signal.
INK_VECTOR_DIM = 6


@dataclass(frozen=True, slots=True)
class ProposalNetConfig:
    # Card embedding lookup table. Row 0 is PAD; rows 1..vocab_size are
    # logical cards in vocab order, shape (vocab_size + 1, embed_dim).
    vocab_size: int
    embed_dim: int = 256
    # Transformer over the multiset of card embeddings.
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ff_dim: int = 1024
    dropout: float = 0.1
    # Ink conditioning head: ink_2hot → R^{d_model}.
    ink_hidden_dim: int = 32
    pad_token_id: int = 0
    # Whether the ``card_embeddings`` tensor is trainable. Default
    # False so the proposal net can be trained independently of the
    # encoder; flip to True for end-to-end fine-tuning.
    freeze_card_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}).",
            )


class InkEmbedding(nn.Module):
    """Projects a 6-dim multi-hot ink vector to the deck token space.

    Two-layer MLP (6 → ink_hidden_dim → d_model). We use GELU + LayerNorm
    to match the CardEncoder style, and initialise the final layer small
    so at step zero the ink contribution is near-zero — the network
    discovers ink conditioning rather than having it dominate the
    untrained card features.
    """

    def __init__(self, cfg: ProposalNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(INK_VECTOR_DIM, cfg.ink_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.ink_hidden_dim),
            nn.Linear(cfg.ink_hidden_dim, cfg.d_model),
        )
        # Shrink the final layer's weights so the ink bias is tiny at
        # init. A factor of 0.1 is enough to keep the cold-start
        # gradient flowing through the card path.
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, std=0.02)
        nn.init.zeros_(last.bias)

    def forward(self, ink_multihot: torch.Tensor) -> torch.Tensor:
        # (B, 6) → (B, d_model)
        out: torch.Tensor = self.net(ink_multihot)
        return out


class ProposalNet(nn.Module):
    """Full proposal net: multiset Transformer + vocab-softmax head."""

    def __init__(
        self,
        cfg: ProposalNetConfig,
        *,
        card_embeddings: torch.Tensor,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        if card_embeddings.shape[0] != cfg.vocab_size + 1:
            raise ValueError(
                f"card_embeddings has {card_embeddings.shape[0]} rows; "
                f"expected vocab_size + 1 = {cfg.vocab_size + 1} (row 0 = PAD).",
            )
        if card_embeddings.shape[1] != cfg.embed_dim:
            raise ValueError(
                f"card_embeddings dim {card_embeddings.shape[1]} "
                f"!= configured embed_dim {cfg.embed_dim}.",
            )
        # Register the frozen table as a parameter only if the caller
        # asked for fine-tuning; otherwise a buffer keeps it out of
        # optimiser param groups and prevents accidental updates.
        if cfg.freeze_card_embeddings:
            self.register_buffer("card_embeddings", card_embeddings.detach().clone())
        else:
            self.card_embeddings = nn.Parameter(card_embeddings.detach().clone())

        # Optional projection when the pretrained encoder's embed_dim
        # doesn't match d_model. Identity path (no extra params) when
        # they already agree so the default DESIGN config is
        # parameter-identical to before this layer was added.
        if cfg.embed_dim == cfg.d_model:
            self.embed_projection: nn.Module = nn.Identity()
        else:
            self.embed_projection = nn.Linear(cfg.embed_dim, cfg.d_model, bias=False)
        self.ink_embed = InkEmbedding(cfg)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm: matches CardEncoder, more stable for small models
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        # 2 · d_model because we concat mean and max pool.
        self.head = nn.Sequential(
            nn.LayerNorm(2 * cfg.d_model),
            nn.Linear(2 * cfg.d_model, cfg.vocab_size + 1),
        )

    def forward(
        self,
        card_ids: torch.Tensor,
        ink_multihot: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits of shape ``(B, vocab_size + 1)``.

        ``card_ids`` is ``(B, N)`` with PAD = 0 on right-padded tail
        positions. ``ink_multihot`` is ``(B, 6)`` float {0, 1}.
        """
        if card_ids.dim() != 2:
            raise ValueError(f"expected card_ids of shape (B, N); got {tuple(card_ids.shape)}")
        if ink_multihot.dim() != 2 or ink_multihot.shape[-1] != INK_VECTOR_DIM:
            raise ValueError(
                f"expected ink_multihot of shape (B, {INK_VECTOR_DIM}); "
                f"got {tuple(ink_multihot.shape)}",
            )

        # Lookup — uses buffer or parameter depending on freeze flag.
        card_vecs = self.card_embeddings[card_ids]  # (B, N, embed_dim)
        # Project to d_model if the encoder's embed_dim is wider/
        # narrower than the Transformer. Identity for the default case.
        card_vecs = self.embed_projection(card_vecs)  # (B, N, d_model)
        ink_vec = self.ink_embed(ink_multihot).unsqueeze(1)  # (B, 1, d_model)
        # Broadcast-add the ink bias across every card slot. PAD slots
        # receive it too — the padding mask excludes them from
        # attention and pooling, so it's inert.
        x = card_vecs + ink_vec

        padding_mask = card_ids == self.cfg.pad_token_id  # True = PAD
        hidden = self.transformer(x, src_key_padding_mask=padding_mask)
        pooled = _masked_mean_max_pool(hidden, padding_mask)
        logits: torch.Tensor = self.head(pooled)
        return logits

    @property
    def gradient_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _masked_mean_max_pool(hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Concat mean and max pools over non-PAD positions.

    Mirrors :func:`models.card_encoder._masked_mean_max_pool` — the
    same reasoning about all-pad rows and max-over-neg-inf applies.
    Kept as a private copy rather than importing so the two modules
    stay independently readable.
    """
    mask = (~padding_mask).unsqueeze(-1).float()
    total = mask.sum(dim=1).clamp(min=1.0)
    mean = (hidden * mask).sum(dim=1) / total
    neg_inf = torch.finfo(hidden.dtype).min
    max_input = hidden.masked_fill(padding_mask.unsqueeze(-1), neg_inf)
    max_val, _ = max_input.max(dim=1)
    max_val = torch.where(padding_mask.all(dim=1, keepdim=True), torch.zeros_like(max_val), max_val)
    return torch.cat([mean, max_val], dim=-1)


def proposal_loss(
    logits: torch.Tensor,
    target_distribution: torch.Tensor,
    *,
    entropy_beta: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-entropy against a soft target − ``β · H(pred)``.

    ``logits`` is ``(B, V)`` raw model output; ``target_distribution`` is
    ``(B, V)`` non-negative and row-sum-1. Returns ``(total_loss, ce,
    entropy)`` for diagnostics. The entropy term is *subtracted* from
    the total, so maximising model entropy reduces total loss.
    """
    if logits.shape != target_distribution.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} != target {tuple(target_distribution.shape)}",
        )
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    # Soft-target cross-entropy: −Σ target · log p.
    ce = -(target_distribution * log_probs).sum(dim=-1).mean()
    # Entropy of the predicted distribution.
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    total = ce - entropy_beta * entropy
    return total, ce, entropy
