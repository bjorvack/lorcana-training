"""MLM + structured-reconstruction heads for card-encoder pretraining.

DESIGN.md: "Two-headed self-supervised pretraining of the card
encoder, before any deck data is involved. Text head: BPE-tokenise
every card's text. Mask 15% of tokens. Predict masked tokens from
context (standard MLM loss). Structured head: mask a random subset
of the structured features and predict them back from the unmasked
parts (denoising autoencoder loss). Combined loss L_pre = L_mlm + L_struct."

This module is the *head* half of that picture — two small
modules that stick on top of the :class:`CardEncoder` and the
prediction + loss functions that turn one forward pass into
``L_pre``. The masking utilities live in
:mod:`lorcana_training.train.masking`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .card_encoder import CardEncoder, CardEncoderConfig


MLM_IGNORE_INDEX = -100  # positions not to score


class MlmHead(nn.Module):
    """Per-position vocab prediction.

    Reads the text encoder's pre-pool hidden states ``(B, T, d_model)``
    and produces logits ``(B, T, vocab_size)``. Weight-tied to the
    encoder's token embedding so the reconstruction signal shares
    parameters with the forward representation — standard recipe,
    shrinks the param count meaningfully on small pools.
    """

    def __init__(self, encoder: CardEncoder) -> None:
        super().__init__()
        cfg = encoder.cfg
        self.cfg = cfg
        self.transform = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        # Tie the output projection to the token embedding matrix.
        self.decoder = nn.Linear(cfg.d_model, cfg.vocab_size, bias=True)
        self.decoder.weight = encoder.text.token_emb.weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.transform(hidden))


class StructReconstructionHead(nn.Module):
    """Reconstruct the full structured-feature vector from the fused card embedding.

    Input is the L2-normalised fused card embedding produced by
    :class:`CardEncoder`. Output is a ``(B, struct_feature_dim)`` vector
    trained under MSE against the original (pre-masking) features.

    Using the fused embedding (not the structured path alone) means the
    text path has to carry enough signal to reconstruct the structure
    when it gets dropped — that's the whole point of the denoising
    objective.
    """

    def __init__(self, cfg: CardEncoderConfig) -> None:
        super().__init__()
        if cfg.struct_feature_dim <= 0:
            raise ValueError("struct_feature_dim must be set on the CardEncoderConfig")
        self.net = nn.Sequential(
            nn.Linear(cfg.encoder_dim, cfg.struct_hidden),
            nn.GELU(),
            nn.Linear(cfg.struct_hidden, cfg.struct_feature_dim),
        )

    def forward(self, card_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(card_embedding)


@dataclass(frozen=True, slots=True)
class PretrainLosses:
    mlm: torch.Tensor
    struct: torch.Tensor
    total: torch.Tensor


def compute_pretrain_loss(
    *,
    encoder: CardEncoder,
    mlm_head: MlmHead,
    struct_head: StructReconstructionHead,
    token_ids: torch.Tensor,
    mlm_labels: torch.Tensor,
    struct_features_masked: torch.Tensor,
    struct_features_target: torch.Tensor,
    struct_weight: float = 1.0,
) -> PretrainLosses:
    """Run a single pretrain forward + loss pass.

    ``token_ids`` are the masked token ids (with [MASK] in ~15% of
    positions); ``mlm_labels`` is ``MLM_IGNORE_INDEX`` everywhere
    except the positions that were masked, where it holds the original
    token id. ``struct_features_masked`` is the input to the encoder
    (with random blocks zeroed out); ``struct_features_target`` is the
    pre-masking original that we train to reconstruct.

    Returns MLM, struct, and total losses as separate tensors so the
    training loop can log them individually.
    """
    # Text path: we need the pre-pool hidden states for MLM.
    hidden, _ = encoder.text.encode_tokens(token_ids)
    mlm_logits = mlm_head(hidden)  # (B, T, V)
    mlm_loss = F.cross_entropy(
        mlm_logits.reshape(-1, mlm_logits.size(-1)),
        mlm_labels.reshape(-1),
        ignore_index=MLM_IGNORE_INDEX,
    )

    # Structured path: forward the encoder end-to-end on the *masked*
    # features, then reconstruct the original.
    card_emb = encoder(token_ids, struct_features_masked)
    reconstruction = struct_head(card_emb)
    struct_loss = F.mse_loss(reconstruction, struct_features_target)

    total = mlm_loss + struct_weight * struct_loss
    return PretrainLosses(mlm=mlm_loss, struct=struct_loss, total=total)
