"""Card encoder: text Transformer + structured MLP fused to R^256.

Two parallel paths map a single card into a fixed-size embedding:

    token ids (B, T)              structured features (B, D_struct)
          │                              │
          ▼                              ▼
    Embedding + pos                  MLP (D_struct -> 128 -> 64)
          │                              │
          ▼                              │
    4x TransformerEncoderLayer           │
          │                              │
          ▼                              │
    mean+max pool → (B, 2*d)             │
          │                              │
          ▼                              │
    MLP (2d -> text_dim=192)             │
          │                              │
          └──────── concat ──────────────┘
                     (B, text_dim + struct_dim)
                          │
                          ▼
                 MLP (-> encoder_dim=256)
                          │
                          ▼
                 L2-normalised card embedding

Design choices:

- Learned positional embeddings rather than sinusoidal — card texts
  are short (most fit in < 64 tokens) and learned positions are
  what the rest of the stack expects too.
- Padding is masked out of attention via ``src_key_padding_mask`` so
  pooling doesn't get polluted by PAD positions.
- Outputs are L2-normalised. Everything downstream (evaluator,
  novelty bonus in lorcana-web) works in cosine space.

The parameter count for the DESIGN.md sizes (d_model=128, heads=4,
layers=4, text_dim=192, struct_dim=64, encoder_dim=256) lands around
1.5 M once you plug in the real ~4 k BPE vocab, or ~5 M with the
DESIGN-target 32 k vocab the tokeniser reserves space for. The
module accepts config via :class:`CardEncoderConfig` so both tests
(tiny configs, quick assertions on shapes) and real training
(DESIGN defaults) use the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class CardEncoderConfig:
    # Text path.
    vocab_size: int
    pad_token_id: int = 0
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ff_dim: int = 512
    max_positions: int = 256
    dropout: float = 0.1
    text_dim: int = 192  # output dim of the text projection head
    # Structured path.
    struct_feature_dim: int = 0  # must be set from feature_schema.json
    struct_hidden: int = 128
    struct_dim: int = 64  # output dim of the struct MLP
    # Fusion head -> card embedding.
    encoder_dim: int = 256


class StructuredEncoder(nn.Module):
    """MLP over the one-hot + normalised scalar card features."""

    def __init__(self, cfg: CardEncoderConfig) -> None:
        super().__init__()
        if cfg.struct_feature_dim <= 0:
            raise ValueError("struct_feature_dim must be set to feature_schema.dim")
        self.net = nn.Sequential(
            nn.Linear(cfg.struct_feature_dim, cfg.struct_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.struct_hidden, cfg.struct_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, struct_feature_dim). Output: (B, struct_dim).
        return self.net(features)


class TextEncoder(nn.Module):
    """Embedding → Transformer → pool → projection. Ignores PAD in attention."""

    def __init__(self, cfg: CardEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.pos_emb = nn.Embedding(cfg.max_positions, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm is more stable for small models
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        # 2 * d_model because we concat mean and max pool.
        self.projection = nn.Sequential(
            nn.LayerNorm(2 * cfg.d_model),
            nn.Linear(2 * cfg.d_model, cfg.text_dim),
            nn.GELU(),
        )

    def encode_tokens(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (hidden, padding_mask). ``hidden`` is ``(B, T, d_model)``,
        ``padding_mask`` is the boolean mask expected by the rest of the
        module (``True`` where padded).
        """
        if token_ids.dim() != 2:
            raise ValueError(f"expected token_ids of shape (B, T); got {tuple(token_ids.shape)}")
        batch_size, seq_len = token_ids.shape
        if seq_len > self.cfg.max_positions:
            raise ValueError(
                f"seq_len={seq_len} exceeds configured max_positions={self.cfg.max_positions}"
            )
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        padding_mask = token_ids == self.cfg.pad_token_id
        hidden = self.transformer(x, src_key_padding_mask=padding_mask)
        return hidden, padding_mask

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden, padding_mask = self.encode_tokens(token_ids)
        pooled = _masked_mean_max_pool(hidden, padding_mask)
        return self.projection(pooled)


class CardEncoder(nn.Module):
    """Full card encoder: text + structured → L2-normalised R^{encoder_dim}."""

    def __init__(self, cfg: CardEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.text = TextEncoder(cfg)
        self.struct = StructuredEncoder(cfg)
        fusion_in = cfg.text_dim + cfg.struct_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, cfg.encoder_dim),
            nn.GELU(),
            nn.Linear(cfg.encoder_dim, cfg.encoder_dim),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        struct_features: torch.Tensor,
        *,
        normalise: bool = True,
    ) -> torch.Tensor:
        text_vec = self.text(token_ids)
        struct_vec = self.struct(struct_features)
        fused = self.fusion(torch.cat([text_vec, struct_vec], dim=-1))
        if normalise:
            fused = F.normalize(fused, dim=-1)
        return fused

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _masked_mean_max_pool(hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """Concatenate mean and max pools over non-pad positions.

    ``padding_mask`` is True where a position is padding (and therefore
    must be excluded from both reductions). Handles all-pad rows by
    falling back to zero pools — callers shouldn't feed those in, but
    letting them through keeps backward passes well-defined during
    testing.
    """
    # hidden: (B, T, d). padding_mask: (B, T). Invert so 1 = real token.
    mask = (~padding_mask).unsqueeze(-1).float()
    total = mask.sum(dim=1).clamp(min=1.0)  # guard against all-pad rows
    mean = (hidden * mask).sum(dim=1) / total
    # max-pool: set pad positions to -inf so they can't win the max.
    neg_inf = torch.finfo(hidden.dtype).min
    max_input = hidden.masked_fill(padding_mask.unsqueeze(-1), neg_inf)
    max_val, _ = max_input.max(dim=1)
    # All-pad rows produce -inf; clamp back to 0 so downstream layers are safe.
    max_val = torch.where(padding_mask.all(dim=1, keepdim=True), torch.zeros_like(max_val), max_val)
    return torch.cat([mean, max_val], dim=-1)
