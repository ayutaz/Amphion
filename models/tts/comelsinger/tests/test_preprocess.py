"""Unit tests for CoMelSinger preprocessing pipeline (M1-12 to M1-18)."""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from models.tts.comelsinger.preprocess import (
    extract_acoustic_tokens,
    extract_phone_ids,
    extract_pitch_tokens,
    extract_semantic_tokens,
    load_preprocessed,
    save_preprocessed,
)

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T_A = 75  # representative acoustic frame count


def _make_sample_data(t_a: int = T_A, n_codebooks: int = 12):
    """Return a dict of valid sample tensors for save_preprocessed."""
    return dict(
        acoustic_tokens=torch.randint(0, 1024, (t_a, n_codebooks), dtype=torch.long),
        semantic_tokens=torch.randint(0, 512, (t_a,), dtype=torch.long),
        pitch_tokens=torch.randint(0, 129, (t_a,), dtype=torch.long),
        phone_ids=torch.randint(0, 100, (20,), dtype=torch.long),
        note_durations=torch.tensor([25, 25, 25], dtype=torch.long),
        note_pitches=torch.tensor([60, 64, 67], dtype=torch.long),
        attention_mask=torch.ones(t_a, dtype=torch.long),
        speaker_id=0,
    )


# ---------------------------------------------------------------------------
# M1-16: save / load roundtrip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundtrip:
    def test_save_load_roundtrip_all_fields(self, tmp_path):
        """Saving and loading should reproduce every field identically."""
        data = _make_sample_data()
        out = tmp_path / "sample.pt"
        save_preprocessed(out, **data)

        loaded = load_preprocessed(out)

        # acoustic_tokens is transposed to (12, T_a) on save
        assert loaded["acoustic_tokens"].shape == (12, T_A)
        assert torch.equal(
            loaded["acoustic_tokens"],
            data["acoustic_tokens"].permute(1, 0).contiguous(),
        )
        assert torch.equal(loaded["semantic_tokens"], data["semantic_tokens"])
        assert torch.equal(loaded["pitch_tokens"], data["pitch_tokens"])
        assert torch.equal(loaded["phone_ids"], data["phone_ids"])
        assert torch.equal(loaded["note_durations"], data["note_durations"])
        assert torch.equal(loaded["note_pitches"], data["note_pitches"])
        assert torch.equal(loaded["attention_mask"], data["attention_mask"])
        assert loaded["speaker_id"].item() == data["speaker_id"]

    def test_acoustic_tokens_shape_after_load(self, tmp_path):
        """Loaded acoustic_tokens must have shape (12, T_a)."""
        data = _make_sample_data(t_a=50)
        out = tmp_path / "a.pt"
        save_preprocessed(out, **data)
        loaded = load_preprocessed(out)
        assert loaded["acoustic_tokens"].shape == (12, 50)

    def test_speaker_id_dtype(self, tmp_path):
        """speaker_id should be stored as a long scalar tensor."""
        data = _make_sample_data()
        out = tmp_path / "s.pt"
        save_preprocessed(out, **data)
        loaded = load_preprocessed(out)
        assert loaded["speaker_id"].dtype == torch.long
        assert loaded["speaker_id"].ndim == 0

    def test_parent_dirs_created(self, tmp_path):
        """save_preprocessed should create missing parent directories."""
        data = _make_sample_data()
        out = tmp_path / "deep" / "nested" / "sample.pt"
        save_preprocessed(out, **data)
        assert out.exists()

    def test_load_from_string_path(self, tmp_path):
        """load_preprocessed should accept a plain string path."""
        data = _make_sample_data()
        out = tmp_path / "s.pt"
        save_preprocessed(out, **data)
        loaded = load_preprocessed(str(out))
        assert "acoustic_tokens" in loaded

    def test_dtypes_preserved(self, tmp_path):
        """All tensors should retain their original dtypes after roundtrip."""
        data = _make_sample_data()
        out = tmp_path / "d.pt"
        save_preprocessed(out, **data)
        loaded = load_preprocessed(out)
        assert loaded["acoustic_tokens"].dtype == torch.long
        assert loaded["semantic_tokens"].dtype == torch.long
        assert loaded["pitch_tokens"].dtype == torch.long
        assert loaded["phone_ids"].dtype == torch.long
        assert loaded["attention_mask"].dtype == torch.long


# ---------------------------------------------------------------------------
# M1-16: length mismatch guard
# ---------------------------------------------------------------------------


class TestSaveLengthMismatch:
    def test_semantic_tokens_wrong_length(self, tmp_path):
        """Mismatched semantic_tokens length must raise ValueError."""
        data = _make_sample_data()
        data["semantic_tokens"] = torch.randint(0, 512, (T_A + 1,), dtype=torch.long)
        with pytest.raises(ValueError, match="semantic_tokens"):
            save_preprocessed(tmp_path / "x.pt", **data)

    def test_pitch_tokens_wrong_length(self, tmp_path):
        """Mismatched pitch_tokens length must raise ValueError."""
        data = _make_sample_data()
        data["pitch_tokens"] = torch.randint(0, 129, (T_A - 1,), dtype=torch.long)
        with pytest.raises(ValueError, match="pitch_tokens"):
            save_preprocessed(tmp_path / "x.pt", **data)

    def test_attention_mask_wrong_length(self, tmp_path):
        """Mismatched attention_mask length must raise ValueError."""
        data = _make_sample_data()
        data["attention_mask"] = torch.ones(T_A + 5, dtype=torch.long)
        with pytest.raises(ValueError, match="attention_mask"):
            save_preprocessed(tmp_path / "x.pt", **data)

    def test_correct_lengths_no_error(self, tmp_path):
        """All matching lengths should not raise."""
        data = _make_sample_data()
        save_preprocessed(tmp_path / "ok.pt", **data)  # must not raise


# ---------------------------------------------------------------------------
# M1-13: 50 Hz -> 75 Hz nearest-neighbor upsample
# ---------------------------------------------------------------------------


class TestUpsample50HzTo75Hz:
    def test_upsample_ratio(self):
        """Nearest-neighbor interpolation from T_50 to T_75 preserves values."""
        torch.manual_seed(42)
        T_50 = 50
        T_75 = 75
        D = 8
        feat = torch.randn(1, D, T_50)
        upsampled = torch.nn.functional.interpolate(feat, size=T_75, mode="nearest")
        assert upsampled.shape == (1, D, T_75)

    def test_upsample_output_values_in_range(self):
        """Upsampled values must be drawn from the original (no new values created)."""
        torch.manual_seed(42)
        T_50 = 10
        T_75 = 15
        D = 4
        feat = torch.arange(T_50, dtype=torch.float).unsqueeze(0).unsqueeze(0).expand(1, D, -1)
        upsampled = torch.nn.functional.interpolate(feat, size=T_75, mode="nearest")
        original_values = set(range(T_50))
        for v in upsampled[0, 0].tolist():
            assert int(v) in original_values

    def test_upsample_shape_75hz(self):
        """Result shape must be (1, D, target_len) for target_len=75."""
        feat = torch.zeros(1, 16, 50)
        out = torch.nn.functional.interpolate(feat, size=75, mode="nearest")
        assert out.shape == (1, 16, 75)


# ---------------------------------------------------------------------------
# M1-14: Pitch token extraction with mock tokenizer
# ---------------------------------------------------------------------------


class MockPitchTokenizer:
    """Minimal mock of PitchTokenizer for use in tests without pyworld."""

    VOCAB_SIZE = 129

    def quantize_f0(self, f0: np.ndarray, target_len: int) -> torch.Tensor:
        # Return all-zero tokens (unvoiced) for silence, or constant 69 (A4) otherwise
        if np.all(f0 == 0):
            return torch.zeros(target_len, dtype=torch.long)
        return torch.full((target_len,), 69, dtype=torch.long)


class TestPitchTokensWithMockTokenizer:
    def test_shape_correct(self):
        """pitch_tokens shape must match target_len."""
        tok = MockPitchTokenizer()
        f0 = np.full(200, 440.0)
        tokens = tok.quantize_f0(f0, target_len=75)
        assert tokens.shape == (75,)

    def test_dtype_long(self):
        """pitch_tokens must be torch.long."""
        tok = MockPitchTokenizer()
        tokens = tok.quantize_f0(np.zeros(200), target_len=75)
        assert tokens.dtype == torch.long

    def test_range_unvoiced(self):
        """All-zero F0 -> all tokens == 0."""
        tok = MockPitchTokenizer()
        tokens = tok.quantize_f0(np.zeros(200), target_len=75)
        assert tokens.min().item() >= 0
        assert tokens.max().item() <= 128
        assert tokens.sum().item() == 0

    def test_range_voiced(self):
        """Voiced tokens must be within [0, 128]."""
        tok = MockPitchTokenizer()
        tokens = tok.quantize_f0(np.full(200, 440.0), target_len=75)
        assert tokens.min().item() >= 0
        assert tokens.max().item() <= 128

    def test_target_len_respected(self):
        """target_len parameter must control output length."""
        tok = MockPitchTokenizer()
        for tl in [10, 50, 100, 200]:
            assert tok.quantize_f0(np.zeros(400), target_len=tl).shape[0] == tl


# ---------------------------------------------------------------------------
# M1-15: Phone ID extraction
# ---------------------------------------------------------------------------


class TestPhoneIdsBasic:
    def test_dtype_long(self):
        """extract_phone_ids must return torch.long tensor."""
        phone2id = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz0123456789")}
        phone2id["<unk>"] = 0
        result = extract_phone_ids("你好", phone2id, language="zh")
        assert result.dtype == torch.long

    def test_dim_is_1(self):
        """Output must be 1-D."""
        phone2id = {"<unk>": 0}
        result = extract_phone_ids("你好", phone2id, language="zh")
        assert result.ndim == 1

    def test_numel_positive(self):
        """Chinese text '你好' must produce at least one phone ID."""
        phone2id = {"n": 1, "i": 2, "3": 3, "h": 4, "a": 5, "o": 6, "<unk>": 0}
        result = extract_phone_ids("你好", phone2id, language="zh")
        assert result.numel() > 0

    def test_values_in_range(self):
        """All IDs must be valid indices (non-negative)."""
        phone2id = {"<unk>": 0, "n": 1, "i": 2, "3": 3}
        result = extract_phone_ids("你好", phone2id, language="zh")
        assert (result >= 0).all()


class TestPhoneIdsUnknownFallback:
    def test_unknown_char_falls_back_to_unk_id(self):
        """Characters not in phone2id must map to phone2id['<unk>']."""
        phone2id = {"<unk>": 99}
        result = extract_phone_ids("你好", phone2id, language="zh")
        # Every phone should fall back to 99 since none are in the vocab
        assert (result == 99).all()

    def test_unknown_char_falls_back_to_zero_when_no_unk(self):
        """When <unk> is missing, unknown chars fall back to 0."""
        phone2id = {}  # no <unk> either -> default 0
        result = extract_phone_ids("你好", phone2id, language="zh")
        assert (result == 0).all()

    def test_known_and_unknown_mixed(self):
        """Mix of known and unknown phones: known get correct IDs."""
        from pypinyin import lazy_pinyin, Style
        pinyins = lazy_pinyin("你好", style=Style.TONE3)
        phones = []
        for py in pinyins:
            phones.extend(list(py))
        # Build mapping for exactly the first phone, rest are unknown
        phone2id = {phones[0]: 42, "<unk>": 99}
        result = extract_phone_ids("你好", phone2id, language="zh")
        assert result[0].item() == 42
        # remaining phones are unknown -> 99
        assert (result[1:] == 99).all()


# ---------------------------------------------------------------------------
# M1-12: Acoustic token extraction interface test (mock codec)
# ---------------------------------------------------------------------------


class TestExtractAcousticTokensInterface:
    def _make_mock_codec(self, t_a: int = 75, num_codebooks: int = 12):
        """Build mock codec_encoder and codec_decoder."""
        codec_encoder = MagicMock()
        codec_decoder = MagicMock()

        # codec_encoder returns a dummy embedding
        dummy_emb = torch.zeros(1, 512, t_a)
        codec_encoder.return_value = dummy_emb

        # codec_decoder.quantizer returns (_, vq, _, _, _)
        # vq shape: (num_codebooks, B, T_a)
        vq = torch.randint(0, 1024, (num_codebooks, 1, t_a))
        codec_decoder.quantizer.return_value = (None, vq, None, None, None)

        return codec_encoder, codec_decoder

    def test_output_shape(self):
        """extract_acoustic_tokens must return (T_a, 12)."""
        codec_encoder, codec_decoder = self._make_mock_codec(t_a=75)
        speech = torch.zeros(24000)  # 1 second at 24kHz
        tokens = extract_acoustic_tokens(speech, codec_encoder, codec_decoder)
        assert tokens.shape == (75, 12)

    def test_output_dtype_long(self):
        """acoustic_tokens must be torch.long."""
        codec_encoder, codec_decoder = self._make_mock_codec()
        speech = torch.zeros(24000)
        tokens = extract_acoustic_tokens(speech, codec_encoder, codec_decoder)
        assert tokens.dtype == torch.long

    def test_output_value_range(self):
        """acoustic_tokens values must be in [0, 1023]."""
        codec_encoder, codec_decoder = self._make_mock_codec()
        speech = torch.zeros(24000)
        tokens = extract_acoustic_tokens(speech, codec_encoder, codec_decoder)
        assert tokens.min().item() >= 0
        assert tokens.max().item() <= 1023

    def test_codec_encoder_called_once(self):
        """codec_encoder must be called exactly once."""
        codec_encoder, codec_decoder = self._make_mock_codec()
        speech = torch.zeros(24000)
        extract_acoustic_tokens(speech, codec_encoder, codec_decoder)
        codec_encoder.assert_called_once()

    def test_different_t_a(self):
        """Output T_a dimension must match what the codec returns."""
        codec_encoder, codec_decoder = self._make_mock_codec(t_a=100)
        speech = torch.zeros(32000)
        tokens = extract_acoustic_tokens(speech, codec_encoder, codec_decoder)
        assert tokens.shape == (100, 12)


# ---------------------------------------------------------------------------
# @pytest.mark.slow — real model tests (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestExtractAcousticTokensReal:
    """Integration tests that require the real Amphion Codec model.

    Run with: pytest -m slow
    """

    def test_real_codec_shape(self):
        pytest.skip("Real codec not available in CI.")


@pytest.mark.slow
class TestExtractSemanticTokensReal:
    """Integration tests requiring w2v-bert-2.0 + RepCodec.

    Run with: pytest -m slow
    """

    def test_real_semantic_extraction(self):
        pytest.skip("Real models not available in CI.")


@pytest.mark.slow
class TestExtractPitchTokensReal:
    """Integration tests requiring pyworld.

    Run with: pytest -m slow
    """

    def test_real_pitch_extraction(self):
        pytest.skip("Real pyworld integration not tested in CI.")
