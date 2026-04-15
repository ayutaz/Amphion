"""SVTModule: Singing Voice Transcription module.

Encoder-only Transformer that predicts frame-level pitch tokens
from acoustic tokens. Used as frozen auxiliary supervision during
S2A training (paper Section III-C).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (L,) long or float tensor of position indices.
        Returns:
            (L, dim) float tensor.
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=positions.device, dtype=torch.float32) * -emb
        )
        emb = positions.float().unsqueeze(-1) * emb.unsqueeze(0)  # (L, half_dim)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (L, dim)


class SVTModule(nn.Module):
    """Encoder-only Transformer: acoustic tokens -> pitch prediction.

    Architecture (paper Section III-C, requirements spec Section 4):
    - 12 codebook embeddings (1024 -> 64 each) -> concat (768) -> linear (512)
    - Sinusoidal positional encoding (512)
    - TransformerEncoder: 4 layers, 8 heads, dim_feedforward=2048, norm_first=True
    - Pitch head: Linear(512, 129)

    Input: acoustic_tokens (B, L, 12) long
    Output: {"logits": (B, L, 129), "probs": (B, L, 129)}
    """

    def __init__(
        self,
        num_codebooks: int = 12,
        codebook_size: int = 1024,
        codebook_embed_dim: int = 64,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        pitch_vocab_size: int = 129,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_embed_dim = codebook_embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pitch_vocab_size = pitch_vocab_size

        # M2-02: Codebook embeddings + input projection
        self.codebook_embs = nn.ModuleList(
            [nn.Embedding(codebook_size, codebook_embed_dim) for _ in range(num_codebooks)]
        )
        self.input_proj = nn.Linear(num_codebooks * codebook_embed_dim, hidden_size)

        # M2-03: Positional encoding + Transformer
        self.pos_enc = SinusoidalPosEmb(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is unsupported with norm_first=True; disable explicitly.
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # M2-04: Pitch prediction head
        self.pitch_head = nn.Linear(hidden_size, pitch_vocab_size)

    def forward(
        self,
        acoustic_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            acoustic_tokens: (B, L, 12) long, values in [0, 1023].
            attention_mask: (B, L) where 1=valid, 0=padding. None means all valid.

        Returns:
            dict with:
                "logits": (B, L, 129) raw logits (pre-softmax)
                "probs":  (B, L, 129) softmax probabilities
        """
        B, L, _ = acoustic_tokens.shape

        # Embed each codebook independently -> concat -> project to hidden_size
        embeds = [self.codebook_embs[i](acoustic_tokens[:, :, i]) for i in range(self.num_codebooks)]
        x = torch.cat(embeds, dim=-1)  # (B, L, num_codebooks * codebook_embed_dim)
        x = self.input_proj(x)  # (B, L, hidden_size)

        # Add positional encoding
        positions = torch.arange(L, device=x.device)
        x = x + self.pos_enc(positions).unsqueeze(0)  # broadcast over batch dim

        # Transformer with padding mask
        # TransformerEncoder expects src_key_padding_mask where True = ignore (padding)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()  # invert: 1=valid -> False=attend

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, L, hidden_size)

        # Pitch prediction
        logits = self.pitch_head(x)  # (B, L, pitch_vocab_size)
        probs = F.softmax(logits, dim=-1)

        return {"logits": logits, "probs": probs}

    # M2-05: Freeze/unfreeze utilities

    def freeze(self) -> None:
        """Freeze all parameters and switch to eval mode.

        Used during S2A training where SVT acts as a frozen auxiliary supervisor
        (StopGrad in paper Section III-C).
        """
        self.requires_grad_(False)
        self.eval()

    def unfreeze(self) -> None:
        """Unfreeze all parameters and switch to train mode."""
        self.requires_grad_(True)
        self.train()
