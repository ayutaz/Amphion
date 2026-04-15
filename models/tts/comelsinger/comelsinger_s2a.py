"""CoMelSinger S2A model — extends MaskGCT_S2A with pitch embedding."""
from __future__ import annotations

import torch
import torch.nn as nn
from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A


class CoMelSinger_S2A(MaskGCT_S2A):
    """MaskGCT_S2A extended with pitch conditioning for singing voice synthesis.

    Adds:
    - pitch_emb: nn.Embedding(129, 1024) for pitch conditioning
    - forward override: cond = cond_emb(cond_code) + pitch_emb(pitch_tokens)
    - get_cond: helper for inference pipeline

    Paper Section III-A, confirmed loss weights:
        lambda_cl=0.5, lambda_scl=1.0, lambda_fcl=0.1, lambda_svt=0.5, lambda_mask=0.3
    """

    def __init__(
        self,
        pitch_vocab_size: int = 129,
        temperature: float = 0.07,
        lambda_cl: float = 0.5,
        lambda_scl: float = 1.0,
        lambda_fcl: float = 0.1,
        lambda_svt: float = 0.5,
        lambda_mask: float = 0.3,
        **kwargs,
    ) -> None:
        # Ensure num_quantizer=12 for CoMelSinger
        kwargs.setdefault("num_quantizer", 12)
        super().__init__(**kwargs)

        # M2-12: Pitch embedding (same dim as hidden_size=1024)
        self.pitch_emb = nn.Embedding(pitch_vocab_size, self.hidden_size)

        # Hyperparameters
        self.pitch_vocab_size = pitch_vocab_size
        self.temperature = temperature
        self.lambda_cl = lambda_cl
        self.lambda_scl = lambda_scl
        self.lambda_fcl = lambda_fcl
        self.lambda_svt = lambda_svt
        self.lambda_mask = lambda_mask

    def forward(self, x0, x_mask, cond_code=None, pitch_tokens=None):
        """Training forward — adds pitch embedding to condition.

        Args:
            x0: (B, T, num_quantizer) acoustic tokens
            x_mask: (B, T) padding mask
            cond_code: (B, T) semantic token codes
            pitch_tokens: (B, T) pitch tokens (optional, None for backward compat)

        Returns:
            Same as MaskGCT_S2A.forward(): logits, mask_layer, final_mask, x0, prompt_len, mask_prob
        """
        # Build condition: semantic + pitch
        cond = self.cond_emb(cond_code)  # (B, T, hidden_size)
        if pitch_tokens is not None:
            cond = cond + self.pitch_emb(pitch_tokens)  # element-wise add

        # Clone cond before passing to compute_loss because loss_t() does
        # in-place addition: cond += layer_emb(mask_layer)
        return self.compute_loss(x0, x_mask, cond.clone())

    def get_cond(self, cond_code, pitch_tokens=None):
        """Build condition tensor for inference (used by reverse_diffusion).

        Args:
            cond_code: (B, T) semantic token codes
            pitch_tokens: (B, T) pitch tokens (optional)

        Returns:
            cond: (B, T, hidden_size)
        """
        cond = self.cond_emb(cond_code)
        if pitch_tokens is not None:
            cond = cond + self.pitch_emb(pitch_tokens)
        return cond
