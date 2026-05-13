"""Per-step Evaluator — pointwise (partial_deck, candidate_card) → plausibility.

Architecture (DESIGN.md §3 "Per-step Evaluator"):

    partial deck (B, N)                     candidate (B,)
         │                                       │
         ▼                                       ▼
    card_embeddings[ids]                card_embeddings[ids]
    (B, N, embed_dim)                   (B, embed_dim)
         │                                       │
         ▼                                       ▼
    [optional] Linear→d_model      MLP(embed_dim → d_model)
         │                                       │
         ▼                                       │
    2-layer Transformer encoder                  │
    d=256, heads=4                               │
         │                                       │
         ▼                                       │
    mean+max pool over non-PAD → (B, 2·d_model)  │
         │                                       │
         └──────── concat ─────────────── (B, 2·d_model + d_model) ─┐
                                                                   ▼
                                               MLP(3·d_model → d_model → 1)
                                                                   │
                                                                   ▼
                                                               sigmoid
                                                        (scalar V ∈ (0, 1))

The "pointwise" part is load-bearing: scoring one (partial, candidate)
pair at a time is what lets the web-side search loop call the
evaluator on every legal next card inside the beam search. A listwise
or pairwise model would bake the candidate set into training and
couldn't be reused at inference against a different legal set.

Design choices:

- **Shared frozen card embeddings** (registered as a buffer when
  ``freeze_card_embeddings=True``) — same trick as :class:`ProposalNet`.
  Keeps the evaluator independently trainable against any encoder
  export and makes checkpoints ~2 MB smaller.
- **2 Transformer layers, 4 heads** (vs. 6 / 8 in the proposal). The
  evaluator only has to recognise "does this candidate fit?"; it
  doesn't need to model a distribution over the whole vocab. Smaller
  net = less overfit on a training set that's only a few thousand
  decks after curriculum-phase expansion.
- **No ink conditioning here.** The partial deck already encodes
  inks through its own card embeddings (ink is a feature every card
  inherits via the CardEncoder). Adding a second ink channel would
  duplicate signal the Transformer already sees.
- **Pre-norm Transformer + LayerNorm on the fusion head** — identical
  stability trick to the proposal net, so small-model training on
  laptops produces consistent numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class EvaluatorConfig:
    # Card embedding table: row 0 = PAD, rows 1..vocab_size = logical
    # cards in vocab order. Shape (vocab_size + 1, embed_dim).
    vocab_size: int
    embed_dim: int = 256
    # Transformer over the partial-deck multiset.
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 512
    dropout: float = 0.1
    pad_token_id: int = 0
    # Hidden dim of the (R^3d → R^d → R^1) fusion head.
    head_hidden_dim: int = 256
    # Whether the card_embeddings tensor is trainable. Default False so
    # this module can be trained against any frozen encoder export.
    freeze_card_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}).",
            )


class Evaluator(nn.Module):
    """Pointwise (partial, candidate) → plausibility discriminator."""

    def __init__(
        self,
        cfg: EvaluatorConfig,
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
        if cfg.freeze_card_embeddings:
            self.register_buffer("card_embeddings", card_embeddings.detach().clone())
        else:
            self.card_embeddings = nn.Parameter(card_embeddings.detach().clone())

        # Optional bridge when embed_dim != d_model — identical story to
        # the proposal net's embed_projection. Identity when they match
        # so the default config is parameter-identical to a version of
        # this class without the hook.
        if cfg.embed_dim == cfg.d_model:
            self.embed_projection: nn.Module = nn.Identity()
        else:
            self.embed_projection = nn.Linear(cfg.embed_dim, cfg.d_model, bias=False)

        # Partial-deck path: shared bridge → 2-layer set Transformer.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.deck_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # Candidate path: 2-layer MLP on the card's own embedding.
        self.candidate_mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Fusion head: concat [deck_mean, deck_max, candidate] → R^1.
        fusion_in = 2 * cfg.d_model + cfg.d_model
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, cfg.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, 1),
        )

    def forward(
        self,
        partial_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits ``(B,)`` — apply sigmoid for probabilities.

        Returning logits (rather than sigmoid probabilities) matches the
        BCE-with-logits loss we train against; wrapping a sigmoid here
        and then re-logging for BCE loses numerical stability in the
        tails.
        """
        if partial_ids.dim() != 2:
            raise ValueError(f"partial_ids must be (B, N); got {tuple(partial_ids.shape)}")
        if candidate_ids.dim() != 1:
            raise ValueError(f"candidate_ids must be (B,); got {tuple(candidate_ids.shape)}")
        if partial_ids.shape[0] != candidate_ids.shape[0]:
            raise ValueError(
                f"batch mismatch: partial_ids={tuple(partial_ids.shape)}, "
                f"candidate_ids={tuple(candidate_ids.shape)}",
            )

        # Embed + project to d_model.
        deck_vecs = self.embed_projection(self.card_embeddings[partial_ids])  # (B, N, d_model)
        cand_vecs = self.embed_projection(self.card_embeddings[candidate_ids])  # (B, d_model)

        # Encode partial deck as an unordered set.
        padding_mask = partial_ids == self.cfg.pad_token_id
        deck_hidden = self.deck_encoder(deck_vecs, src_key_padding_mask=padding_mask)
        deck_pooled = _masked_mean_max_pool(deck_hidden, padding_mask)  # (B, 2·d_model)

        # Candidate MLP.
        cand_vec = self.candidate_mlp(cand_vecs)  # (B, d_model)

        fused = torch.cat([deck_pooled, cand_vec], dim=-1)  # (B, 3·d_model)
        logits: torch.Tensor = self.head(fused).squeeze(-1)  # (B,)
        return logits

    @property
    def gradient_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _masked_mean_max_pool(hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Concat mean and max pools over non-PAD positions. Same contract as
    :func:`lorcana_training.models.proposal._masked_mean_max_pool`."""
    mask = (~padding_mask).unsqueeze(-1).float()
    total = mask.sum(dim=1).clamp(min=1.0)
    mean = (hidden * mask).sum(dim=1) / total
    neg_inf = torch.finfo(hidden.dtype).min
    max_input = hidden.masked_fill(padding_mask.unsqueeze(-1), neg_inf)
    max_val, _ = max_input.max(dim=1)
    max_val = torch.where(padding_mask.all(dim=1, keepdim=True), torch.zeros_like(max_val), max_val)
    return torch.cat([mean, max_val], dim=-1)


def evaluator_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """BCE-with-logits loss. ``labels`` is a float tensor ∈ {0.0, 1.0}."""
    if logits.shape != labels.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)} != labels {tuple(labels.shape)}",
        )
    return F.binary_cross_entropy_with_logits(logits, labels.float())
