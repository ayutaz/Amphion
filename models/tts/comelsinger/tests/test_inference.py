"""Unit tests for CoMelSinger inference pipeline (M4-01 to M4-04)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from models.tts.comelsinger.comelsinger_inference import (
    CoMelSingerInferencePipeline,
)
from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer


# ---------------------------------------------------------------------------
# TestCoMelSingerPipeline
# ---------------------------------------------------------------------------


class TestCoMelSingerPipeline:
    """Tests for CoMelSingerInferencePipeline construction and basic behavior."""

    def test_init_default(self) -> None:
        """Pipeline can be created with no arguments (all models None)."""
        pipeline = CoMelSingerInferencePipeline()
        assert pipeline.device == torch.device("cpu")
        assert pipeline.s2a_model_1layer is None
        assert pipeline.s2a_model_full is None
        assert pipeline.t2s_model is None
        assert pipeline.codec_encoder is None
        assert pipeline.codec_decoder is None
        assert pipeline.semantic_model is None
        assert pipeline.semantic_codec is None

    def test_init_with_device(self) -> None:
        """Pipeline respects custom device string."""
        pipeline = CoMelSingerInferencePipeline(device="cpu")
        assert pipeline.device == torch.device("cpu")

    def test_pitch_tokenizer_present(self) -> None:
        """Pipeline creates a default PitchTokenizer if none provided."""
        pipeline = CoMelSingerInferencePipeline()
        assert isinstance(pipeline.pitch_tokenizer, PitchTokenizer)

    def test_pitch_tokenizer_custom(self) -> None:
        """Pipeline accepts a custom PitchTokenizer."""
        custom_tok = PitchTokenizer(vocab_size=64, encodec_fps=50.0)
        pipeline = CoMelSingerInferencePipeline(pitch_tokenizer=custom_tok)
        assert pipeline.pitch_tokenizer is custom_tok
        assert pipeline.pitch_tokenizer.vocab_size == 64

    def test_synthesize_raises_without_models(self) -> None:
        """synthesize() raises NotImplementedError when models are missing."""
        pipeline = CoMelSingerInferencePipeline()
        with pytest.raises(NotImplementedError, match="Missing"):
            pipeline.synthesize(
                prompt_wav_path="dummy.wav",
                lyrics="test",
                pitch_sequence=[60],
                note_durations=[1.0],
            )

    def test_n_timesteps_default_12_elements(self) -> None:
        """Default n_timesteps_s2a has exactly 12 elements."""
        assert len(CoMelSingerInferencePipeline.DEFAULT_N_TIMESTEPS_S2A) == 12
        assert CoMelSingerInferencePipeline.DEFAULT_N_TIMESTEPS_S2A[0] == 25
        assert CoMelSingerInferencePipeline.DEFAULT_N_TIMESTEPS_S2A[1] == 10
        # Remaining should all be 1
        for ts in CoMelSingerInferencePipeline.DEFAULT_N_TIMESTEPS_S2A[2:]:
            assert ts == 1

    def test_check_models_ready_all_missing(self) -> None:
        """_check_models_ready lists all missing models."""
        pipeline = CoMelSingerInferencePipeline()
        with pytest.raises(NotImplementedError) as exc_info:
            pipeline._check_models_ready()
        msg = str(exc_info.value)
        assert "s2a_model_1layer" in msg
        assert "s2a_model_full" in msg
        assert "t2s_model" in msg
        assert "codec_encoder" in msg
        assert "codec_decoder" in msg
        assert "semantic_model" in msg
        assert "semantic_codec" in msg

    def test_check_models_ready_partial(self) -> None:
        """_check_models_ready only reports actually missing models."""
        pipeline = CoMelSingerInferencePipeline(
            t2s_model=MagicMock(),
            codec_encoder=MagicMock(),
            codec_decoder=MagicMock(),
        )
        with pytest.raises(NotImplementedError) as exc_info:
            pipeline._check_models_ready()
        msg = str(exc_info.value)
        assert "s2a_model_1layer" in msg
        assert "t2s_model" not in msg
        assert "codec_encoder" not in msg

    def test_from_pretrained_raises(self) -> None:
        """from_pretrained raises NotImplementedError until M0 is complete."""
        with pytest.raises(NotImplementedError, match="pretrained model downloads"):
            CoMelSingerInferencePipeline.from_pretrained("dummy_config.yaml")


# ---------------------------------------------------------------------------
# TestTokenizeScore
# ---------------------------------------------------------------------------


class TestTokenizeScore:
    """Tests for the tokenize_score helper method."""

    def test_basic_tokenization(self) -> None:
        """tokenize_score produces correct length pitch tokens."""
        pipeline = CoMelSingerInferencePipeline()
        pitch_tokens, frame_counts = pipeline.tokenize_score(
            pitch_sequence=[60, 62, 64, 67],
            note_durations=[0.5, 0.5, 0.5, 0.5],
        )
        # Total 2.0s * 75 fps = 150 frames
        expected_len = int(2.0 * 75.0)
        assert pitch_tokens.shape[0] == expected_len
        assert frame_counts.sum().item() == expected_len

    def test_single_note(self) -> None:
        """Single note produces correct tokens."""
        pipeline = CoMelSingerInferencePipeline()
        pitch_tokens, frame_counts = pipeline.tokenize_score(
            pitch_sequence=[69],
            note_durations=[1.0],
        )
        assert pitch_tokens.shape[0] == 75  # 1.0s * 75fps
        assert (pitch_tokens == 69).all()

    def test_frame_counts_match_notes(self) -> None:
        """frame_counts has one entry per note."""
        pipeline = CoMelSingerInferencePipeline()
        n_notes = 8
        _, frame_counts = pipeline.tokenize_score(
            pitch_sequence=[60 + i for i in range(n_notes)],
            note_durations=[0.25] * n_notes,
        )
        assert frame_counts.shape[0] == n_notes


# ---------------------------------------------------------------------------
# TestLoadLoraWeights
# ---------------------------------------------------------------------------


class TestLoadLoraWeights:
    """Tests for LoRA weight loading."""

    def test_load_lora_skips_if_no_model(self) -> None:
        """load_lora_weights does nothing if s2a models are None."""
        pipeline = CoMelSingerInferencePipeline()
        # Should not raise even though models are None
        # Direct test: if model is None, load_lora_weights should not attempt PeftModel import
        pipeline.s2a_model_1layer = None
        pipeline.s2a_model_full = None
        # load_lora_weights with both models None should be a no-op
        pipeline.load_lora_weights("/dummy/path", model_type="both")
        assert pipeline.s2a_model_1layer is None
        assert pipeline.s2a_model_full is None

    def test_load_lora_invalid_model_type(self) -> None:
        """load_lora_weights raises ValueError for invalid model_type."""
        pipeline = CoMelSingerInferencePipeline()
        with pytest.raises(ValueError, match="model_type must be"):
            pipeline.load_lora_weights("/dummy/path", model_type="invalid")


# ---------------------------------------------------------------------------
# TestRunInference
# ---------------------------------------------------------------------------


class TestRunInference:
    """Tests for the batch inference CLI (run_inference.py)."""

    def test_cli_help(self) -> None:
        """CLI --help runs successfully."""
        result = subprocess.run(
            [sys.executable, "-m", "models.tts.comelsinger.run_inference", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[4]),
        )
        assert result.returncode == 0
        assert "--testset" in result.stdout
        assert "--output_dir" in result.stdout
        assert "--config" in result.stdout
        assert "--device" in result.stdout
        assert "--ablation_tag" in result.stdout

    def test_cli_missing_required_args(self) -> None:
        """CLI fails cleanly without required arguments."""
        result = subprocess.run(
            [sys.executable, "-m", "models.tts.comelsinger.run_inference"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[4]),
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# TestLoadTestset
# ---------------------------------------------------------------------------


class TestLoadTestset:
    """Tests for testset loading utility."""

    def test_load_json_array(self, tmp_path: Path) -> None:
        """Loads a standard JSON array testset."""
        from models.tts.comelsinger.run_inference import load_testset

        testset = [
            {
                "id": "s1",
                "wav_path": "ref.wav",
                "text": "hello",
                "pitch_sequence": [60],
                "note_durations": [1.0],
            }
        ]
        testset_path = tmp_path / "test.json"
        with open(testset_path, "w") as f:
            json.dump(testset, f)

        result = load_testset(testset_path)
        assert len(result) == 1
        assert result[0]["id"] == "s1"

    def test_load_jsonl(self, tmp_path: Path) -> None:
        """Loads a JSONL format testset."""
        from models.tts.comelsinger.run_inference import load_testset

        lines = [
            json.dumps({"id": "s1", "wav_path": "a.wav", "text": "t1",
                        "pitch_sequence": [60], "note_durations": [1.0]}),
            json.dumps({"id": "s2", "wav_path": "b.wav", "text": "t2",
                        "pitch_sequence": [62], "note_durations": [0.5]}),
        ]
        testset_path = tmp_path / "test.jsonl"
        with open(testset_path, "w") as f:
            f.write("\n".join(lines))

        result = load_testset(testset_path)
        assert len(result) == 2

    def test_load_missing_file(self) -> None:
        """Raises FileNotFoundError for missing testset."""
        from models.tts.comelsinger.run_inference import load_testset

        with pytest.raises(FileNotFoundError):
            load_testset(Path("/nonexistent/testset.json"))


# ---------------------------------------------------------------------------
# TestValidateSample
# ---------------------------------------------------------------------------


class TestValidateSample:
    """Tests for sample validation."""

    def test_valid_sample(self) -> None:
        """Valid sample passes validation."""
        from models.tts.comelsinger.run_inference import validate_sample

        sample = {
            "wav_path": "ref.wav",
            "text": "hello",
            "pitch_sequence": [60, 62],
            "note_durations": [0.5, 0.5],
        }
        # Should not raise
        validate_sample(sample, 0)

    def test_missing_fields(self) -> None:
        """Missing required fields raises ValueError."""
        from models.tts.comelsinger.run_inference import validate_sample

        with pytest.raises(ValueError, match="missing required fields"):
            validate_sample({"wav_path": "x.wav"}, 0)

    def test_length_mismatch(self) -> None:
        """Mismatched pitch/duration lengths raises ValueError."""
        from models.tts.comelsinger.run_inference import validate_sample

        sample = {
            "wav_path": "ref.wav",
            "text": "hello",
            "pitch_sequence": [60, 62],
            "note_durations": [0.5],
        }
        with pytest.raises(ValueError, match="pitch_sequence length"):
            validate_sample(sample, 0)


# ---------------------------------------------------------------------------
# TestRunInferenceFn
# ---------------------------------------------------------------------------


class TestRunInferenceFn:
    """Tests for the run_inference function."""

    def test_handles_not_implemented(self, tmp_path: Path) -> None:
        """run_inference gracefully handles NotImplementedError from pipeline."""
        from models.tts.comelsinger.run_inference import run_inference

        pipeline = CoMelSingerInferencePipeline()
        samples = [
            {
                "id": "test_001",
                "wav_path": "dummy.wav",
                "text": "hello",
                "pitch_sequence": [60],
                "note_durations": [1.0],
            }
        ]

        metadata = run_inference(
            pipeline=pipeline,
            samples=samples,
            output_dir=tmp_path / "output",
        )

        assert metadata["n_total"] == 1
        assert metadata["n_ok"] == 0
        assert metadata["n_skip"] == 1
        assert metadata["samples"][0]["status"] in ("skip", "error")

    def test_handles_validation_error(self, tmp_path: Path) -> None:
        """run_inference skips samples with validation errors."""
        from models.tts.comelsinger.run_inference import run_inference

        pipeline = CoMelSingerInferencePipeline()
        samples = [
            {
                "id": "bad_sample",
                "wav_path": "x.wav",
                "text": "hello",
                "pitch_sequence": [60, 62],
                "note_durations": [0.5],  # length mismatch
            }
        ]

        metadata = run_inference(
            pipeline=pipeline,
            samples=samples,
            output_dir=tmp_path / "output",
        )

        assert metadata["n_skip"] == 1
        assert "error" in metadata["samples"][0]
