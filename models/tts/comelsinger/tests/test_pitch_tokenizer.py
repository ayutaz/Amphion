"""Unit tests for PitchTokenizer (M1-01 to M1-06)."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch

from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tok() -> PitchTokenizer:
    """Default PitchTokenizer instance."""
    return PitchTokenizer()


# ---------------------------------------------------------------------------
# TestPitchTokenizerInit
# ---------------------------------------------------------------------------


class TestPitchTokenizerInit:
    """Tests for class constants and __init__."""

    def test_class_constant_vocab_size(self) -> None:
        assert PitchTokenizer.VOCAB_SIZE == 129

    def test_class_constant_encodec_fps(self) -> None:
        assert PitchTokenizer.ENCODEC_FPS == 75.0

    def test_class_constant_f0_fps(self) -> None:
        assert PitchTokenizer.F0_FPS == 200.0

    def test_instance_vocab_size(self, tok: PitchTokenizer) -> None:
        assert tok.vocab_size == 129

    def test_instance_encodec_fps(self, tok: PitchTokenizer) -> None:
        assert tok.encodec_fps == 75.0

    def test_instance_f0_fps(self, tok: PitchTokenizer) -> None:
        assert tok.f0_fps == 200.0

    def test_custom_params(self) -> None:
        custom = PitchTokenizer(vocab_size=64, encodec_fps=50.0, f0_fps=100.0)
        assert custom.vocab_size == 64
        assert custom.encodec_fps == 50.0
        assert custom.f0_fps == 100.0


# ---------------------------------------------------------------------------
# TestQuantizeF0
# ---------------------------------------------------------------------------


class TestQuantizeF0:
    """Tests for quantize_f0 method."""

    def test_a4_440hz(self, tok: PitchTokenizer) -> None:
        """440 Hz (A4) should map to MIDI token 69."""
        result = tok.quantize_f0(np.array([440.0] * 3), 1)
        assert result[0].item() == 69

    def test_unvoiced(self, tok: PitchTokenizer) -> None:
        """0.0 Hz (unvoiced) should map to token 0."""
        result = tok.quantize_f0(np.array([0.0] * 3), 1)
        assert result[0].item() == 0

    def test_c4(self, tok: PitchTokenizer) -> None:
        """Middle C (261.63 Hz) should map to MIDI token 60."""
        result = tok.quantize_f0(np.array([261.63] * 3), 1)
        assert result[0].item() == 60

    def test_output_length(self, tok: PitchTokenizer) -> None:
        """Output shape must equal target_len."""
        f0 = np.zeros(200)
        result = tok.quantize_f0(f0, 75)
        assert result.shape == (75,)

    def test_mixed_voiced_unvoiced(self, tok: PitchTokenizer) -> None:
        """First half voiced, second half unvoiced."""
        torch.manual_seed(42)
        # 200 frames at 200Hz = 1 second; 75 frames at 75Hz = 1 second
        # First 100 F0 frames voiced, last 100 unvoiced
        f0 = np.concatenate([np.full(100, 440.0), np.zeros(100)])
        result = tok.quantize_f0(f0, 75)
        assert result.shape == (75,)
        # Roughly first half should be 69 (voiced), second half 0 (unvoiced)
        # Boundary may shift slightly due to frame rounding
        assert result[0].item() == 69
        assert result[-1].item() == 0

    def test_clamp_low(self, tok: PitchTokenizer) -> None:
        """Extremely low frequency should clamp to token 1."""
        # 1 Hz is well below MIDI 1 (~8.18 Hz)
        result = tok.quantize_f0(np.array([1.0] * 3), 1)
        assert result[0].item() == 1

    def test_clamp_high(self, tok: PitchTokenizer) -> None:
        """Extremely high frequency should clamp to token 128."""
        # 20000 Hz is well above MIDI 128 (~12543.9 Hz)
        result = tok.quantize_f0(np.array([20000.0] * 3), 1)
        assert result[0].item() == 128

    def test_all_unvoiced(self, tok: PitchTokenizer) -> None:
        """All-zero F0 should produce all-zero tokens."""
        f0 = np.zeros(400)
        result = tok.quantize_f0(f0, 75)
        assert result.sum().item() == 0
        assert result.shape == (75,)

    def test_dtype_long(self, tok: PitchTokenizer) -> None:
        """Output dtype should be torch.long."""
        result = tok.quantize_f0(np.array([440.0]), 1)
        assert result.dtype == torch.long

    def test_value_range(self, tok: PitchTokenizer) -> None:
        """All output tokens must be in [0, 128]."""
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        f0 = rng.uniform(50, 2000, 300).astype(np.float64)
        result = tok.quantize_f0(f0, 100)
        assert result.min().item() >= 0
        assert result.max().item() <= 128

    def test_target_len_larger_than_f0(self, tok: PitchTokenizer) -> None:
        """Edge case: target_len > len(f0) * fps_ratio should not crash."""
        f0 = np.full(3, 440.0)  # very short F0
        result = tok.quantize_f0(f0, 10)
        assert result.shape == (10,)
        assert result.min().item() >= 0
        assert result.max().item() <= 128

    def test_880hz_a5(self, tok: PitchTokenizer) -> None:
        """880 Hz (A5) should map to MIDI token 81."""
        result = tok.quantize_f0(np.array([880.0] * 3), 1)
        assert result[0].item() == 81

    def test_220hz_a3(self, tok: PitchTokenizer) -> None:
        """220 Hz (A3) should map to MIDI token 57."""
        result = tok.quantize_f0(np.array([220.0] * 3), 1)
        assert result[0].item() == 57

    def test_quantize_f0_negative_values(self, tok: PitchTokenizer) -> None:
        """Negative F0 values should be treated as unvoiced."""
        result = tok.quantize_f0(np.array([-1.0, -100.0]), 1)
        assert result[0].item() == 0


# ---------------------------------------------------------------------------
# TestTokenizeScore
# ---------------------------------------------------------------------------


class TestTokenizeScore:
    """Tests for tokenize_score method."""

    def test_basic_two_notes(self, tok: PitchTokenizer) -> None:
        """Two equal-duration notes in 10 frames -> each gets 5 frames."""
        tokens, fc = tok.tokenize_score([60, 64], [1, 1], 10)
        assert tokens.shape[0] == 10
        assert fc[0].item() == 5
        assert fc[1].item() == 5

    def test_sum_guarantee(self, tok: PitchTokenizer) -> None:
        """sum(frame_counts) must equal target_len for various configurations."""
        torch.manual_seed(42)
        rng = random.Random(42)
        for _ in range(200):
            S = rng.randint(1, 20)
            midi = [rng.randint(48, 84) for _ in range(S)]
            dur = [rng.randint(1, 8) for _ in range(S)]
            L = rng.randint(S, 300)
            _, fc = tok.tokenize_score(midi, dur, L)
            assert fc.sum().item() == L, (
                f"sum mismatch: {fc.sum().item()} != {L} (S={S}, dur={dur}, L={L})"
            )

    def test_min_one_frame(self, tok: PitchTokenizer) -> None:
        """Each note must receive at least 1 frame."""
        torch.manual_seed(42)
        rng = random.Random(42)
        for _ in range(200):
            S = rng.randint(1, 20)
            midi = [rng.randint(48, 84) for _ in range(S)]
            dur = [rng.randint(1, 8) for _ in range(S)]
            L = rng.randint(S, 300)
            _, fc = tok.tokenize_score(midi, dur, L)
            assert (fc >= 1).all(), (
                f"frame_count < 1 found: {fc.tolist()} (S={S}, L={L})"
            )

    def test_unvoiced_rest(self, tok: PitchTokenizer) -> None:
        """MIDI 0 (rest) should produce token 0 (unvoiced)."""
        tokens, fc = tok.tokenize_score([0, 60], [1, 1], 10)
        # First 5 frames should all be 0 (rest)
        assert tokens[:fc[0].item()].sum().item() == 0

    def test_unequal_durations(self, tok: PitchTokenizer) -> None:
        """Unequal durations: [1.0, 3.0] with target=8 -> fc=[2, 6]."""
        tokens, fc = tok.tokenize_score([60, 64], [1, 3], 8)
        assert tokens.shape[0] == 8
        assert fc.sum().item() == 8
        assert (fc >= 1).all()
        # Note 1 has duration 1/4 -> 2 frames, note 2 has 3/4 -> 6 frames
        assert fc[0].item() == 2
        assert fc[1].item() == 6

    def test_single_note(self, tok: PitchTokenizer) -> None:
        """Single note: all frames should have the same token."""
        tokens, fc = tok.tokenize_score([69], [1], 10)
        assert tokens.shape[0] == 10
        assert fc[0].item() == 10
        assert (tokens == 69).all()

    def test_token_values_correct(self, tok: PitchTokenizer) -> None:
        """Token values in output should match note_midi values."""
        tokens, fc = tok.tokenize_score([60, 64, 67], [1, 1, 1], 12)
        assert fc[0].item() == 4
        assert fc[1].item() == 4
        assert fc[2].item() == 4
        assert (tokens[:4] == 60).all()
        assert (tokens[4:8] == 64).all()
        assert (tokens[8:] == 67).all()

    def test_target_len_equals_num_notes(self, tok: PitchTokenizer) -> None:
        """Extreme case: target_len == S, each note gets exactly 1 frame."""
        tokens, fc = tok.tokenize_score([60, 62, 64], [1, 1, 1], 3)
        assert tokens.shape[0] == 3
        assert fc.sum().item() == 3
        assert (fc >= 1).all()

    def test_midi_clamp(self, tok: PitchTokenizer) -> None:
        """Out-of-range MIDI values should be clamped to [0, 128]."""
        tokens, _ = tok.tokenize_score([200, -5], [1, 1], 10)
        assert tokens.max().item() <= 128
        assert tokens.min().item() >= 0

    def test_output_dtype(self, tok: PitchTokenizer) -> None:
        """Output dtypes should be torch.long."""
        tokens, fc = tok.tokenize_score([60, 64], [1, 1], 10)
        assert tokens.dtype == torch.long
        assert fc.dtype == torch.long

    def test_longer_note_gets_more_frames(self, tok: PitchTokenizer) -> None:
        """A longer note should receive at least as many frames as a shorter one."""
        _, fc = tok.tokenize_score([60, 64], [3, 1], 8)
        assert fc[0].item() >= fc[1].item()


# ---------------------------------------------------------------------------
# TestComputeSoftLabelMatrix
# ---------------------------------------------------------------------------


class TestComputeSoftLabelMatrix:
    """Tests for compute_soft_label_matrix method."""

    def test_same_pitch(self, tok: PitchTokenizer) -> None:
        """All frames with the same voiced pitch -> all entries 1.0."""
        m_p = torch.tensor([60, 60, 60], dtype=torch.long)
        Y = tok.compute_soft_label_matrix(m_p)
        assert Y.shape == (3, 3)
        assert (Y == 1.0).all()

    def test_different_pitch(self, tok: PitchTokenizer) -> None:
        """Different pitches -> only diagonal (self-pairs) are 1.0."""
        m_p = torch.tensor([60, 64], dtype=torch.long)
        Y = tok.compute_soft_label_matrix(m_p)
        assert Y[0, 0].item() == 1.0
        assert Y[1, 1].item() == 1.0
        assert Y[0, 1].item() == 0.0
        assert Y[1, 0].item() == 0.0

    def test_unvoiced_not_positive(self, tok: PitchTokenizer) -> None:
        """Unvoiced frames (token=0) should never be positive examples."""
        m_p = torch.tensor([0, 0], dtype=torch.long)
        Y = tok.compute_soft_label_matrix(m_p)
        assert Y.sum().item() == 0.0

    def test_symmetry(self, tok: PitchTokenizer) -> None:
        """Y must be symmetric: Y == Y.T."""
        torch.manual_seed(42)
        m_p = torch.randint(0, 129, (20,))
        Y = tok.compute_soft_label_matrix(m_p)
        assert (Y == Y.T).all()

    def test_dtype_float32(self, tok: PitchTokenizer) -> None:
        """Output dtype must be torch.float32."""
        m_p = torch.tensor([60, 64], dtype=torch.long)
        Y = tok.compute_soft_label_matrix(m_p)
        assert Y.dtype == torch.float32

    def test_values_binary(self, tok: PitchTokenizer) -> None:
        """All values in Y must be exactly 0.0 or 1.0."""
        torch.manual_seed(42)
        m_p = torch.randint(0, 129, (30,))
        Y = tok.compute_soft_label_matrix(m_p)
        unique_values = Y.unique()
        for v in unique_values:
            assert v.item() in {0.0, 1.0}, f"Unexpected value: {v.item()}"

    def test_shape(self, tok: PitchTokenizer) -> None:
        """Output shape must be (L, L)."""
        m_p = torch.randint(0, 129, (50,))
        Y = tok.compute_soft_label_matrix(m_p)
        assert Y.shape == (50, 50)

    def test_voiced_unvoiced_cross_is_zero(self, tok: PitchTokenizer) -> None:
        """Cross pair (voiced, unvoiced) should be 0.0."""
        m_p = torch.tensor([69, 0], dtype=torch.long)
        Y = tok.compute_soft_label_matrix(m_p)
        assert Y[0, 1].item() == 0.0
        assert Y[1, 0].item() == 0.0

    def test_all_unvoiced_all_zero(self, tok: PitchTokenizer) -> None:
        """All-unvoiced input -> all-zero matrix."""
        m_p = torch.zeros(10, dtype=torch.long)
        Y = tok.compute_soft_label_matrix(m_p)
        assert Y.sum().item() == 0.0

    def test_all_same_voiced_full_ones(self, tok: PitchTokenizer) -> None:
        """All same voiced pitch -> full ones matrix."""
        m_p = torch.full((5,), 69, dtype=torch.long)
        Y = tok.compute_soft_label_matrix(m_p)
        assert Y.sum().item() == 25.0


# ---------------------------------------------------------------------------
# TestDecodeTokenToFreq
# ---------------------------------------------------------------------------


class TestDecodeTokenToFreq:
    """Tests for decode_token_to_freq method."""

    def test_a4(self, tok: PitchTokenizer) -> None:
        """Token 69 (A4) should decode to 440.0 Hz."""
        assert abs(tok.decode_token_to_freq(69) - 440.0) < 1e-6

    def test_unvoiced(self, tok: PitchTokenizer) -> None:
        """Token 0 (unvoiced) should decode to 0.0 Hz."""
        assert tok.decode_token_to_freq(0) == 0.0

    def test_octave(self, tok: PitchTokenizer) -> None:
        """Token 81 (A5) should be exactly 2x token 69 (A4)."""
        f_a4 = tok.decode_token_to_freq(69)
        f_a5 = tok.decode_token_to_freq(81)
        assert abs(f_a5 / f_a4 - 2.0) < 1e-9

    def test_invalid_high(self, tok: PitchTokenizer) -> None:
        """Token > 128 should raise ValueError."""
        with pytest.raises(ValueError):
            tok.decode_token_to_freq(129)

    def test_invalid_low(self, tok: PitchTokenizer) -> None:
        """Token < 0 should raise ValueError."""
        with pytest.raises(ValueError):
            tok.decode_token_to_freq(-1)

    def test_monotone(self, tok: PitchTokenizer) -> None:
        """Frequencies should be strictly increasing from token 1 to 128."""
        freqs = [tok.decode_token_to_freq(t) for t in range(1, 129)]
        for i in range(len(freqs) - 1):
            assert freqs[i] < freqs[i + 1], (
                f"Not monotone at token {i + 1}: {freqs[i]} >= {freqs[i + 1]}"
            )

    def test_all_finite_nonnegative(self, tok: PitchTokenizer) -> None:
        """All tokens [0, 128] should produce finite, non-negative frequencies."""
        for t in range(129):
            f = tok.decode_token_to_freq(t)
            assert f >= 0.0, f"Negative freq at token {t}: {f}"
            assert math.isfinite(f), f"Non-finite freq at token {t}: {f}"

    def test_a3(self, tok: PitchTokenizer) -> None:
        """Token 57 (A3) should decode to 220.0 Hz."""
        assert abs(tok.decode_token_to_freq(57) - 220.0) < 1e-6

    def test_a5(self, tok: PitchTokenizer) -> None:
        """Token 81 (A5) should decode to 880.0 Hz."""
        assert abs(tok.decode_token_to_freq(81) - 880.0) < 1e-6

    def test_boundary_token_128(self, tok: PitchTokenizer) -> None:
        """Token 128 should return a valid frequency ~12543.9 Hz."""
        f = tok.decode_token_to_freq(128)
        assert math.isfinite(f)
        # MIDI 128 = 440 * 2^((128-69)/12) ≈ 13289.75 Hz
        expected = 440.0 * (2.0 ** ((128 - 69) / 12.0))
        assert abs(f - expected) < 0.01


# ---------------------------------------------------------------------------
# TestRoundTrip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Round-trip tests: decode_token_to_freq -> quantize_f0 -> same token."""

    def test_all_voiced_tokens(self, tok: PitchTokenizer) -> None:
        """All voiced tokens (1-128) should survive the decode -> quantize round-trip."""
        # For each token, decode to freq, feed 3 copies to quantize_f0, expect same token
        for token in range(1, 129):
            freq = tok.decode_token_to_freq(token)
            # Use 3 samples to ensure the median is stable
            f0 = np.array([freq, freq, freq])
            recovered = tok.quantize_f0(f0, 1)[0].item()
            assert recovered == token, (
                f"Round-trip failed for token {token}: got {recovered} "
                f"(freq={freq:.4f} Hz)"
            )

    def test_unvoiced_round_trip(self, tok: PitchTokenizer) -> None:
        """Token 0 (unvoiced) decodes to 0.0 Hz and re-encodes to 0."""
        freq = tok.decode_token_to_freq(0)
        assert freq == 0.0
        result = tok.quantize_f0(np.array([0.0, 0.0, 0.0]), 1)
        assert result[0].item() == 0


# ---------------------------------------------------------------------------
# TestE2E
# ---------------------------------------------------------------------------


class TestE2E:
    """End-to-end pipeline tests combining multiple methods."""

    def test_tokenize_score_then_soft_label(self, tok: PitchTokenizer) -> None:
        """tokenize_score output can be directly fed into compute_soft_label_matrix."""
        tokens, _ = tok.tokenize_score([60, 64, 67], [1, 1, 1], 12)
        Y = tok.compute_soft_label_matrix(tokens)
        assert Y.shape == (12, 12)
        assert Y.dtype == torch.float32
        assert (Y == Y.T).all()

    def test_pipeline_consistency(self, tok: PitchTokenizer) -> None:
        """Full pipeline: score -> tokens -> soft_label -> values are correct."""
        # Two identical notes: soft label between same-pitch frames should be 1.0
        tokens, fc = tok.tokenize_score([69, 69], [1, 1], 6)
        Y = tok.compute_soft_label_matrix(tokens)
        # All frames have pitch 69, so entire matrix should be 1.0
        assert Y.sum().item() == 36.0  # 6x6 all-ones

    def test_pipeline_with_rest(self, tok: PitchTokenizer) -> None:
        """Rest note (0) in score: corresponding frames should not be positive pairs."""
        tokens, fc = tok.tokenize_score([0, 69], [1, 1], 10)
        Y = tok.compute_soft_label_matrix(tokens)
        # Rest frames should have Y[i, j] = 0.0 with all other frames
        n_rest = fc[0].item()
        # Check rest rows/cols are all zero
        assert Y[:n_rest, :].sum().item() == 0.0
        assert Y[:, :n_rest].sum().item() == 0.0
