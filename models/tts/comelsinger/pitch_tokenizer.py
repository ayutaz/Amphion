"""PitchTokenizer: F0 continuous values or MIDI scores to frame-aligned discrete pitch tokens."""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch


class PitchTokenizer:
    """Generate frame-aligned discrete pitch tokens from F0 continuous values or MIDI scores."""

    VOCAB_SIZE: int = 129        # 0: unvoiced, 1-128: MIDI note 1-128
    ENCODEC_FPS: float = 75.0    # Amphion Codec frame rate (24kHz / 320 samples)
    F0_FPS: float = 200.0        # pyworld DIO frame_period=5ms

    def __init__(
        self,
        vocab_size: int = 129,
        encodec_fps: float = 75.0,
        f0_fps: float = 200.0,
    ) -> None:
        self.vocab_size = vocab_size
        self.encodec_fps = encodec_fps
        self.f0_fps = f0_fps

    def quantize_f0(self, f0: np.ndarray, target_len: int) -> torch.LongTensor:
        """Convert F0 array (T_f0,) Hz to pitch token sequence (L,).

        For each EnCodec frame, takes the median of voiced F0 samples in the
        corresponding F0 time window, converts to MIDI, and clamps to [1, 128].
        Unvoiced frames (f0 == 0 or NaN) produce token 0.

        Args:
            f0: F0 array (T_f0,) in Hz. Unvoiced frames are 0.0.
            target_len: Output token sequence length (= acoustic token frame count T_a).

        Returns:
            tokens: shape (target_len,), dtype torch.long, values in [0, 128].
        """
        tokens = torch.zeros(target_len, dtype=torch.long)

        for i in range(target_len):
            start = round(i * self.f0_fps / self.encodec_fps)
            end = round((i + 1) * self.f0_fps / self.encodec_fps)
            # Clamp end to f0 length; ensure at least start+1 for edge cases
            end = max(end, start + 1)
            end = min(end, len(f0))

            if start >= len(f0):
                tokens[i] = 0
                continue

            frame = f0[start:end]
            # Filter out unvoiced (0.0) and NaN frames
            voiced = frame[(frame > 0) & np.isfinite(frame)]
            if len(voiced) == 0:
                tokens[i] = 0  # unvoiced
                continue

            f_median = float(np.median(voiced))
            midi = round(12.0 * math.log2(f_median / 440.0) + 69)
            tokens[i] = max(1, min(128, midi))

        return tokens

    def tokenize_score(
        self,
        note_midi: List[int],
        duration_symbol: List[int],
        target_len: int,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Convert a musical score to (pitch token sequence (L,), frame count sequence (S,)).

        Uses Largest Remainder Method to distribute frames among notes.
        Guarantees sum(frame_counts) == target_len and all(frame_counts >= 1).

        Args:
            note_midi: MIDI note number sequence (S,). Rest notes are 0.
            duration_symbol: Relative duration values (S,). Positive integers.
            target_len: Target output length L (= acoustic token frame count T_a).

        Returns:
            tokens: shape (L,), dtype torch.long.
            frame_counts: shape (S,), dtype torch.long.
                          sum(frame_counts) == target_len, all(frame_counts >= 1).
        """
        S = len(note_midi)
        D = sum(duration_symbol)
        assert D > 0, "Sum of duration_symbol must be positive"

        # Compute continuous frame allocation for each note
        a_d_raw = [d / D * target_len for d in duration_symbol]

        # Largest Remainder Method for integer allocation
        a_d = [int(x) for x in a_d_raw]
        remainders = [(a_d_raw[i] - a_d[i], i) for i in range(S)]
        deficit = target_len - sum(a_d)
        # Assign +1 to notes with the largest remainders
        remainders.sort(key=lambda x: -x[0])
        for _, idx in remainders[:deficit]:
            a_d[idx] += 1

        # Guarantee at least 1 frame per note
        for i in range(S):
            if a_d[i] < 1:
                a_d[i] = 1

        # Correct for any surplus introduced by the minimum-1 guarantee
        diff = sum(a_d) - target_len
        if diff > 0:
            # Remove surplus frames from notes with the most frames first
            order = sorted(range(S), key=lambda i: -a_d[i])
            for idx in order:
                if diff <= 0:
                    break
                if a_d[idx] > 1:
                    subtract = min(a_d[idx] - 1, diff)
                    a_d[idx] -= subtract
                    diff -= subtract

        # Build pitch token sequence by repeating each MIDI note for its frame count
        token_list: List[int] = []
        for midi, count in zip(note_midi, a_d):
            # Clamp to [0, 128]; rest (0) is preserved as unvoiced token 0
            tok = max(0, min(128, midi))
            token_list.extend([tok] * count)

        tokens = torch.tensor(token_list, dtype=torch.long)
        frame_counts = torch.tensor(a_d, dtype=torch.long)

        assert tokens.shape[0] == target_len, (
            f"tokens length {tokens.shape[0]} != target_len {target_len}"
        )
        assert frame_counts.sum().item() == target_len, (
            f"frame_counts sum {frame_counts.sum().item()} != target_len {target_len}"
        )
        assert (frame_counts >= 1).all(), "frame_counts contains values < 1"

        return tokens, frame_counts

    def compute_soft_label_matrix(
        self, m_p: torch.LongTensor
    ) -> torch.FloatTensor:
        """Compute soft label matrix for Frame-level Contrastive Learning (FCL).

        Y[i,j] = 1.0 if frames i and j share the same pitch AND both are voiced.
        Y[i,j] = 0.0 otherwise (unvoiced frames are never positive examples).

        Args:
            m_p: Pitch token sequence (L,), dtype=long.
                 0 = unvoiced, 1-128 = MIDI note number.

        Returns:
            Y: Soft label matrix (L, L), dtype=float32.
               Values are in {0.0, 1.0}.
        """
        m_p = m_p.long()
        voiced_mask = (m_p != 0)                                          # (L,)
        voiced_2d = voiced_mask.unsqueeze(1) & voiced_mask.unsqueeze(0)   # (L, L)
        same_pitch = (m_p.unsqueeze(1) == m_p.unsqueeze(0))               # (L, L)
        Y = (voiced_2d & same_pitch).float()                               # (L, L)
        return Y

    def decode_token_to_freq(self, token: int) -> float:
        """Convert a discrete pitch token back to frequency in Hz.

        Inverse of quantize_f0. Note that due to MIDI rounding, the round-trip
        frequency may differ slightly from the original F0.

        Args:
            token: Pitch token, int, valid range [0, 128].
                   0 -> unvoiced -> 0.0 Hz.
                   1-128 -> MIDI note number -> corresponding frequency Hz.

        Returns:
            freq: Frequency in Hz (float). Returns 0.0 for token 0.

        Raises:
            ValueError: If token is outside [0, 128].
        """
        token = int(token)
        if token < 0 or token > 128:
            raise ValueError(
                f"token must be in [0, 128], got {token}"
            )
        if token == 0:
            return 0.0
        return 440.0 * (2.0 ** ((token - 69) / 12.0))
