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
import safetensors
import safetensors.torch
import torch

from models.tts.comelsinger.comelsinger_inference import (
    CoMelSingerInferencePipeline,
)
from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer


def _make_from_pretrained_mocks():
    """Build a dict of mock objects suitable for patching from_pretrained imports.

    Returns dict mapping patch target -> mock, plus a helper to apply them.
    Since maskgct_utils imports g2p which requires espeak, we must prevent
    its import by mocking the entire import chain at the function level.
    """
    mock_load_config = MagicMock(return_value=MagicMock())
    mock_build_acoustic = MagicMock(return_value=(MagicMock(), MagicMock()))
    mock_build_semantic = MagicMock(return_value=MagicMock())

    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    mock_ctx.keys.return_value = ["some.weight"]
    mock_ctx.get_tensor.return_value = torch.zeros(1)

    mock_safe_open = MagicMock(return_value=mock_ctx)
    mock_load_model = MagicMock()
    mock_torch_load = MagicMock(return_value={
        "mean": torch.zeros(1024),
        "var": torch.ones(1024),
    })

    mock_t2s_cls = MagicMock(return_value=MagicMock())
    mock_s2a_cls = MagicMock(return_value=MagicMock())

    mock_w2v_bert_cls = MagicMock()
    mock_w2v_bert_cls.from_pretrained.return_value = MagicMock()

    # Build a fake maskgct_utils module to inject into sys.modules
    mock_maskgct_utils = MagicMock()
    mock_maskgct_utils.build_acoustic_codec = mock_build_acoustic
    mock_maskgct_utils.build_semantic_codec = mock_build_semantic

    # Build a fake utils.util module
    mock_utils_util = MagicMock()
    mock_utils_util.load_config = mock_load_config

    # Build fake safetensors modules (PyO3 can only init once per process)
    mock_safetensors = MagicMock()
    mock_safetensors.safe_open = mock_safe_open
    mock_safetensors_torch = MagicMock()
    mock_safetensors_torch.load_model = mock_load_model
    mock_safetensors.torch = mock_safetensors_torch

    return {
        "load_config": mock_load_config,
        "build_acoustic": mock_build_acoustic,
        "build_semantic": mock_build_semantic,
        "safe_open": mock_safe_open,
        "safe_ctx": mock_ctx,
        "load_model": mock_load_model,
        "torch_load": mock_torch_load,
        "t2s_cls": mock_t2s_cls,
        "s2a_cls": mock_s2a_cls,
        "w2v_bert_cls": mock_w2v_bert_cls,
        "maskgct_utils_mod": mock_maskgct_utils,
        "utils_util_mod": mock_utils_util,
        "safetensors_mod": mock_safetensors,
        "safetensors_torch_mod": mock_safetensors_torch,
    }


def _sys_modules_dict(m):
    """Return a dict suitable for patch.dict('sys.modules', ...).

    Only mock modules whose import chain triggers unavailable system tools
    (espeak via g2p). maskgct_utils imports g2p_generation which needs espeak.
    """
    return {
        "models.tts.maskgct.maskgct_utils": m["maskgct_utils_mod"],
        "models.tts.maskgct.g2p": MagicMock(),
        "models.tts.maskgct.g2p.g2p_generation": MagicMock(),
    }


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

    def test_from_pretrained_loads_all_models(self) -> None:
        """from_pretrained correctly calls loaders and returns a pipeline."""
        m = _make_from_pretrained_mocks()

        with (
            patch.dict("sys.modules", _sys_modules_dict(m)),
            patch("utils.util.load_config", m["load_config"]),
            patch.object(safetensors, "safe_open", m["safe_open"]),
            patch.object(safetensors.torch, "load_model", m["load_model"]),
            patch("torch.load", m["torch_load"]),
            patch("models.tts.maskgct.maskgct_t2s.MaskGCT_T2S", m["t2s_cls"]),
            patch("models.tts.comelsinger.comelsinger_s2a.CoMelSinger_S2A", m["s2a_cls"]),
            patch("transformers.Wav2Vec2BertModel", m["w2v_bert_cls"]),
            tempfile.TemporaryDirectory() as tmpdir,
        ):
            ckpt_dir = Path(tmpdir)
            pipeline = CoMelSingerInferencePipeline.from_pretrained(
                ckpt_dir, device="cpu"
            )

        assert isinstance(pipeline, CoMelSingerInferencePipeline)
        assert pipeline.t2s_model is not None
        assert pipeline.s2a_model_1layer is not None
        assert pipeline.s2a_model_full is not None
        m["load_config"].assert_called_once()
        m["build_acoustic"].assert_called_once()
        m["build_semantic"].assert_called_once()
        m["w2v_bert_cls"].from_pretrained.assert_called_once_with("facebook/w2v-bert-2.0")


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


# ---------------------------------------------------------------------------
# TestFromPretrained
# ---------------------------------------------------------------------------


class TestFromPretrained:
    """Tests for from_pretrained via subprocess to avoid module state pollution."""

    def test_correct_checkpoint_paths_used(self) -> None:
        """from_pretrained passes correct paths for each model component."""
        # Run in subprocess to avoid sys.modules contamination between tests
        script = '''
import sys, tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import torch

# Build mock for maskgct_utils
mock_utils = MagicMock()
mock_utils.build_acoustic_codec.return_value = (MagicMock(), MagicMock())
mock_utils.build_semantic_codec.return_value = MagicMock()

mock_load_config = MagicMock(return_value=MagicMock())

mock_ctx = MagicMock()
mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
mock_ctx.__exit__ = MagicMock(return_value=False)
mock_ctx.keys.return_value = []
mock_safe_open = MagicMock(return_value=mock_ctx)
mock_load_model = MagicMock()
mock_torch_load = MagicMock(return_value={
    "mean": torch.zeros(1024), "var": torch.ones(1024),
})
mock_t2s = MagicMock(return_value=MagicMock())
mock_s2a = MagicMock(return_value=MagicMock())
mock_w2v = MagicMock()
mock_w2v.from_pretrained.return_value = MagicMock()

import safetensors, safetensors.torch
with (
    patch.dict("sys.modules", {
        "models.tts.maskgct.maskgct_utils": mock_utils,
        "models.tts.maskgct.g2p": MagicMock(),
        "models.tts.maskgct.g2p.g2p_generation": MagicMock(),
    }),
    patch("utils.util.load_config", mock_load_config),
    patch.object(safetensors, "safe_open", mock_safe_open),
    patch.object(safetensors.torch, "load_model", mock_load_model),
    patch("torch.load", mock_torch_load),
    patch("models.tts.maskgct.maskgct_t2s.MaskGCT_T2S", mock_t2s),
    patch("models.tts.comelsinger.comelsinger_s2a.CoMelSinger_S2A", mock_s2a),
    patch("transformers.Wav2Vec2BertModel", mock_w2v),
    tempfile.TemporaryDirectory() as tmpdir,
):
    from models.tts.comelsinger.comelsinger_inference import CoMelSingerInferencePipeline
    ckpt_dir = Path(tmpdir)
    pipeline = CoMelSingerInferencePipeline.from_pretrained(ckpt_dir, device="cpu")

assert isinstance(pipeline, CoMelSingerInferencePipeline)
assert pipeline.t2s_model is not None
assert pipeline.s2a_model_1layer is not None

cfg_arg = mock_load_config.call_args[0][0]
assert cfg_arg.endswith("config/maskgct.json"), f"config path: {cfg_arg}"

stats_arg = mock_torch_load.call_args[0][0]
assert stats_arg.endswith("wav2vec2bert_stats.pt"), f"stats path: {stats_arg}"

assert mock_safe_open.call_count == 4, f"safe_open called {mock_safe_open.call_count} times"

print("PASS")
'''
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[4]),
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout

    def test_lora_loaded_when_path_given(self) -> None:
        """from_pretrained calls load_lora_weights when lora_path is provided."""
        script = '''
import sys, tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import torch

mock_utils = MagicMock()
mock_utils.build_acoustic_codec.return_value = (MagicMock(), MagicMock())
mock_utils.build_semantic_codec.return_value = MagicMock()

mock_ctx = MagicMock()
mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
mock_ctx.__exit__ = MagicMock(return_value=False)
mock_ctx.keys.return_value = []

import safetensors, safetensors.torch
with (
    patch.dict("sys.modules", {
        "models.tts.maskgct.maskgct_utils": mock_utils,
        "models.tts.maskgct.g2p": MagicMock(),
        "models.tts.maskgct.g2p.g2p_generation": MagicMock(),
    }),
    patch("utils.util.load_config", MagicMock(return_value=MagicMock())),
    patch.object(safetensors, "safe_open", MagicMock(return_value=mock_ctx)),
    patch.object(safetensors.torch, "load_model", MagicMock()),
    patch("torch.load", MagicMock(return_value={
        "mean": torch.zeros(1024), "var": torch.ones(1024),
    })),
    patch("models.tts.maskgct.maskgct_t2s.MaskGCT_T2S", MagicMock(return_value=MagicMock())),
    patch("models.tts.comelsinger.comelsinger_s2a.CoMelSinger_S2A", MagicMock(return_value=MagicMock())),
    patch("transformers.Wav2Vec2BertModel", MagicMock(**{"from_pretrained.return_value": MagicMock()})),
):
    from models.tts.comelsinger.comelsinger_inference import CoMelSingerInferencePipeline
    with patch.object(CoMelSingerInferencePipeline, "load_lora_weights") as mock_lora:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = CoMelSingerInferencePipeline.from_pretrained(
                Path(tmpdir), lora_path="/fake/lora", device="cpu"
            )
        mock_lora.assert_called_once_with(Path("/fake/lora"))

print("PASS")
'''
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[4]),
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout


# ---------------------------------------------------------------------------
# TestLoadModels
# ---------------------------------------------------------------------------


class TestLoadModels:
    """Tests for run_preprocess.load_models via subprocess."""

    def test_returns_dict_with_required_keys(self) -> None:
        """load_models returns a dict with all required keys."""
        script = '''
import sys, types
from unittest.mock import MagicMock, patch
import torch

mock_utils = MagicMock()
mock_utils.build_acoustic_codec.return_value = (MagicMock(), MagicMock())
mock_utils.build_semantic_codec.return_value = MagicMock()

mock_ctx = MagicMock()
mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
mock_ctx.__exit__ = MagicMock(return_value=False)
mock_ctx.keys.return_value = []

mock_w2v = MagicMock()
mock_w2v.from_pretrained.return_value = MagicMock()

# Create a mock transformers module with Wav2Vec2BertModel
mock_transformers = types.ModuleType("transformers")
mock_transformers.Wav2Vec2BertModel = mock_w2v

import safetensors, safetensors.torch
with (
    patch.dict("sys.modules", {
        "models.tts.maskgct.maskgct_utils": mock_utils,
        "models.tts.maskgct.g2p": MagicMock(),
        "models.tts.maskgct.g2p.g2p_generation": MagicMock(),
        "transformers": mock_transformers,
    }),
    patch("utils.util.load_config", MagicMock(return_value=MagicMock())),
    patch.object(safetensors, "safe_open", MagicMock(return_value=mock_ctx)),
    patch.object(safetensors.torch, "load_model", MagicMock()),
    patch("torch.load", MagicMock(return_value={
        "mean": torch.zeros(1024), "var": torch.ones(1024),
    })),
):
    from models.tts.comelsinger.run_preprocess import load_models
    result = load_models(torch.device("cpu"))

expected = {"codec_encoder", "codec_decoder", "w2v_bert_model",
            "semantic_codec", "semantic_mean", "semantic_std", "pitch_tokenizer"}
assert isinstance(result, dict)
assert set(result.keys()) == expected, f"keys: {set(result.keys())}"

print("PASS")
'''
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[4]),
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout

    def test_semantic_std_is_sqrt_of_var(self) -> None:
        """semantic_std should be sqrt(var), not raw var."""
        script = '''
import sys, types
from unittest.mock import MagicMock, patch
import torch

mock_utils = MagicMock()
mock_utils.build_acoustic_codec.return_value = (MagicMock(), MagicMock())
mock_utils.build_semantic_codec.return_value = MagicMock()

mock_ctx = MagicMock()
mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
mock_ctx.__exit__ = MagicMock(return_value=False)
mock_ctx.keys.return_value = []

mock_w2v = MagicMock()
mock_w2v.from_pretrained.return_value = MagicMock()

test_var = torch.full((1024,), 4.0)

mock_transformers = types.ModuleType("transformers")
mock_transformers.Wav2Vec2BertModel = mock_w2v

import safetensors, safetensors.torch
with (
    patch.dict("sys.modules", {
        "models.tts.maskgct.maskgct_utils": mock_utils,
        "models.tts.maskgct.g2p": MagicMock(),
        "models.tts.maskgct.g2p.g2p_generation": MagicMock(),
        "transformers": mock_transformers,
    }),
    patch("utils.util.load_config", MagicMock(return_value=MagicMock())),
    patch.object(safetensors, "safe_open", MagicMock(return_value=mock_ctx)),
    patch.object(safetensors.torch, "load_model", MagicMock()),
    patch("torch.load", MagicMock(return_value={
        "mean": torch.zeros(1024), "var": test_var,
    })),
):
    from models.tts.comelsinger.run_preprocess import load_models
    result = load_models(torch.device("cpu"))

expected_std = torch.full((1024,), 2.0)
assert torch.allclose(result["semantic_std"], expected_std), (
    f"Expected std=2.0, got {result['semantic_std'][:5]}"
)

print("PASS")
'''
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[4]),
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout
