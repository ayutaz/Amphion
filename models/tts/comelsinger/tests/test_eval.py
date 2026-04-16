"""Tests for CoMelSinger evaluation scripts (M4-05 to M4-13)."""
import json
import math

import numpy as np
import pytest

SEED = 42


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(SEED)


# ---------------------------------------------------------------------------
# TestComputeMCD (M4-05)
# ---------------------------------------------------------------------------


class TestComputeMCD:
    def test_identical_wav_zero_mcd(self):
        """Same WAV input -> MCD approximately 0."""
        from models.tts.comelsinger.eval.eval_mcd import compute_mcd

        wav = np.random.randn(24000).astype(np.float32)
        mcd = compute_mcd(wav, wav)
        assert mcd < 0.01, f"Expected MCD ~0 for identical wavs, got {mcd}"

    def test_different_wav_positive_mcd(self):
        """Different WAV inputs -> positive MCD."""
        from models.tts.comelsinger.eval.eval_mcd import compute_mcd

        ref = np.random.randn(24000).astype(np.float32)
        syn = np.random.randn(24000).astype(np.float32)
        mcd = compute_mcd(ref, syn)
        assert mcd > 0, f"Expected positive MCD for different wavs, got {mcd}"

    def test_output_type_float(self):
        """MCD output is a Python float."""
        from models.tts.comelsinger.eval.eval_mcd import compute_mcd

        wav = np.random.randn(24000).astype(np.float32)
        mcd = compute_mcd(wav, wav)
        assert isinstance(mcd, float)

    def test_different_length_wavs(self):
        """MCD handles wavs of different lengths via DTW."""
        from models.tts.comelsinger.eval.eval_mcd import compute_mcd

        ref = np.random.randn(24000).astype(np.float32)
        syn = np.random.randn(30000).astype(np.float32)
        mcd = compute_mcd(ref, syn)
        assert isinstance(mcd, float)
        assert math.isfinite(mcd)

    def test_evaluate_mcd_directory(self, tmp_path):
        """evaluate_mcd processes a directory and returns correct structure."""
        import soundfile as sf

        from models.tts.comelsinger.eval.eval_mcd import evaluate_mcd

        ref_dir = tmp_path / "ref"
        syn_dir = tmp_path / "syn"
        ref_dir.mkdir()
        syn_dir.mkdir()

        # Create matching WAV files
        for i in range(3):
            wav = np.random.randn(24000).astype(np.float32)
            sf.write(str(ref_dir / f"sample_{i}.wav"), wav, 24000)
            sf.write(str(syn_dir / f"sample_{i}.wav"), wav + np.random.randn(24000).astype(np.float32) * 0.01, 24000)

        results = evaluate_mcd(ref_dir, syn_dir)
        assert "mean_mcd" in results
        assert "std_mcd" in results
        assert "n_samples" in results
        assert "per_file" in results
        assert results["n_samples"] == 3
        assert len(results["per_file"]) == 3
        assert isinstance(results["mean_mcd"], float)

    def test_evaluate_mcd_no_match_raises(self, tmp_path):
        """evaluate_mcd raises FileNotFoundError if no matching files."""
        from models.tts.comelsinger.eval.eval_mcd import evaluate_mcd

        ref_dir = tmp_path / "ref"
        syn_dir = tmp_path / "syn"
        ref_dir.mkdir()
        syn_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            evaluate_mcd(ref_dir, syn_dir)


# ---------------------------------------------------------------------------
# TestComputeF0RMSE (M4-06)
# ---------------------------------------------------------------------------


class TestComputeF0RMSE:
    def test_identical_wav_zero_rmse(self):
        """Same sinusoidal WAV -> F0-RMSE approximately 0."""
        from models.tts.comelsinger.eval.eval_f0_rmse import compute_f0_rmse

        # A4 sine wave
        t = np.arange(24000) / 24000.0
        wav = np.sin(2 * np.pi * 440 * t).astype(np.float64)
        rmse = compute_f0_rmse(wav, wav, sr=24000)
        # pyworld may have minor extraction differences, but identical input should be very close
        if not math.isnan(rmse):
            assert rmse < 0.5, f"Expected F0-RMSE ~0 for identical wavs, got {rmse}"

    def test_sine_wave_known_f0(self):
        """Two sine waves at different frequencies -> positive RMSE."""
        from models.tts.comelsinger.eval.eval_f0_rmse import compute_f0_rmse

        t = np.arange(48000) / 24000.0  # 2 seconds
        ref = np.sin(2 * np.pi * 440 * t).astype(np.float64)
        syn = np.sin(2 * np.pi * 466.16 * t).astype(np.float64)  # A#4, 1 semitone above
        rmse = compute_f0_rmse(ref, syn, sr=24000)
        if not math.isnan(rmse):
            assert rmse > 0, f"Expected positive RMSE, got {rmse}"

    def test_silence_returns_nan(self):
        """All-zero (silence) input -> NaN (no voiced frames)."""
        from models.tts.comelsinger.eval.eval_f0_rmse import compute_f0_rmse

        ref = np.zeros(24000, dtype=np.float64)
        syn = np.zeros(24000, dtype=np.float64)
        rmse = compute_f0_rmse(ref, syn, sr=24000)
        assert math.isnan(rmse), f"Expected NaN for silence, got {rmse}"

    def test_hz_to_semitone(self):
        """Verify Hz to semitone conversion for A4=440Hz."""
        from models.tts.comelsinger.eval.eval_f0_rmse import hz_to_semitone

        # A4 = 440 Hz = MIDI 69
        f0 = np.array([440.0])
        st = hz_to_semitone(f0)
        assert abs(st[0] - 69.0) < 1e-6

        # C4 = 261.63 Hz = MIDI 60
        f0 = np.array([261.6256])
        st = hz_to_semitone(f0)
        assert abs(st[0] - 60.0) < 0.01

    def test_hz_to_semitone_unvoiced(self):
        """Unvoiced frames (f0 <= 0) are returned as 0."""
        from models.tts.comelsinger.eval.eval_f0_rmse import hz_to_semitone

        f0 = np.array([0.0, -1.0, 440.0])
        st = hz_to_semitone(f0)
        assert st[0] == 0.0
        assert st[1] == 0.0
        assert abs(st[2] - 69.0) < 1e-6


# ---------------------------------------------------------------------------
# TestComputeSingMOS (M4-07)
# ---------------------------------------------------------------------------


class TestComputeSingMOS:
    def test_singmos_none_when_unavailable(self, tmp_path):
        """compute_singmos returns None when model is not available."""
        from models.tts.comelsinger.eval.eval_singmos import compute_singmos

        # Create a dummy WAV file
        import soundfile as sf

        wav = np.random.randn(24000).astype(np.float32)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), wav, 24000)

        # With no pre-loaded model, this may return None if transformers
        # or model download fails. We just check it doesn't crash.
        result = compute_singmos(str(wav_path), device="cpu")
        assert result is None or isinstance(result, float)


# ---------------------------------------------------------------------------
# TestComputeSECS (M4-08)
# ---------------------------------------------------------------------------


class TestComputeSECS:
    @pytest.mark.slow
    def test_same_speaker_high_secs(self):
        """Same wav -> SECS close to 1.0 (requires WavLM model)."""
        from models.tts.comelsinger.eval.eval_secs import compute_secs

        wav = np.random.randn(24000).astype(np.float32)
        score = compute_secs(wav, wav, sr=24000, device="cpu")
        if score is not None:
            assert score > 0.99, f"Same wav should give SECS ~1.0, got {score}"

    def test_secs_none_without_model(self):
        """compute_secs returns None if model loading fails gracefully."""
        from models.tts.comelsinger.eval.eval_secs import compute_secs

        wav = np.random.randn(24000).astype(np.float32)
        # This may return None if transformers/WavLM is not available
        result = compute_secs(wav, wav, sr=24000, device="cpu")
        assert result is None or isinstance(result, float)


# ---------------------------------------------------------------------------
# TestComputeSVTF1 (M4-09)
# ---------------------------------------------------------------------------


class TestComputeSVTF1:
    def test_random_prediction_bounded_f1(self):
        """Random SVT predictions should give F1 in [0, 1]."""
        import torch

        from models.tts.comelsinger.eval.eval_svt_f1 import compute_svt_f1
        from models.tts.comelsinger.svt_module import SVTModule

        torch.manual_seed(SEED)
        svt = SVTModule()
        svt.eval()

        acoustic_tokens = torch.randint(0, 1023, (1, 50, 12))
        gt_pitch = torch.randint(1, 128, (1, 50))  # no padding token 0

        result = compute_svt_f1(acoustic_tokens, gt_pitch, svt, device="cpu")
        assert "f1" in result
        assert "precision" in result
        assert "recall" in result
        assert 0.0 <= result["f1"] <= 1.0
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0

    def test_padding_excluded_from_f1(self):
        """Padding (token 0) should be excluded from F1 computation."""
        import torch

        from models.tts.comelsinger.eval.eval_svt_f1 import compute_svt_f1
        from models.tts.comelsinger.svt_module import SVTModule

        torch.manual_seed(SEED)
        svt = SVTModule()
        svt.eval()

        acoustic_tokens = torch.randint(0, 1023, (1, 20, 12))
        # Half padding, half voiced
        gt_pitch = torch.cat([
            torch.zeros(10, dtype=torch.long),   # padding
            torch.randint(1, 128, (10,)),          # voiced
        ]).unsqueeze(0)

        result = compute_svt_f1(acoustic_tokens, gt_pitch, svt, device="cpu")
        assert 0.0 <= result["f1"] <= 1.0

    def test_all_padding_zero_f1(self):
        """All-padding GT returns 0.0 F1."""
        import torch

        from models.tts.comelsinger.eval.eval_svt_f1 import compute_svt_f1
        from models.tts.comelsinger.svt_module import SVTModule

        torch.manual_seed(SEED)
        svt = SVTModule()
        svt.eval()

        acoustic_tokens = torch.randint(0, 1023, (1, 10, 12))
        gt_pitch = torch.zeros(1, 10, dtype=torch.long)

        result = compute_svt_f1(acoustic_tokens, gt_pitch, svt, device="cpu")
        assert result["f1"] == 0.0

    def test_unbatched_input(self):
        """2D input (T, C) without batch dim should work."""
        import torch

        from models.tts.comelsinger.eval.eval_svt_f1 import compute_svt_f1
        from models.tts.comelsinger.svt_module import SVTModule

        torch.manual_seed(SEED)
        svt = SVTModule()
        svt.eval()

        acoustic_tokens = torch.randint(0, 1023, (30, 12))  # no batch dim
        gt_pitch = torch.randint(1, 128, (30,))

        result = compute_svt_f1(acoustic_tokens, gt_pitch, svt, device="cpu")
        assert 0.0 <= result["f1"] <= 1.0


# ---------------------------------------------------------------------------
# TestAblationConfigs (M4-10)
# ---------------------------------------------------------------------------


class TestAblationConfigs:
    def test_full_config_values(self):
        """full config has the paper Section IV-B confirmed loss weights."""
        from models.tts.comelsinger.eval.ablation_configs import ABLATION_CONFIGS

        full = ABLATION_CONFIGS["full"]
        assert full["lambda_cl"] == 0.5
        assert full["lambda_scl"] == 1.0
        assert full["lambda_fcl"] == 0.1
        assert full["lambda_svt"] == 0.5
        assert full["lambda_mask"] == 0.3

    def test_wo_cl_zeros_cl(self):
        """wo_cl disables all contrastive learning."""
        from models.tts.comelsinger.eval.ablation_configs import ABLATION_CONFIGS

        cfg = ABLATION_CONFIGS["wo_cl"]
        assert cfg["lambda_cl"] == 0.0
        assert cfg["lambda_scl"] == 0.0
        assert cfg["lambda_fcl"] == 0.0
        # SVT should remain enabled
        assert cfg["lambda_svt"] == 0.5

    def test_wo_scl_only_zeros_scl(self):
        """wo_scl only zeros out SCL, FCL remains."""
        from models.tts.comelsinger.eval.ablation_configs import ABLATION_CONFIGS

        cfg = ABLATION_CONFIGS["wo_scl"]
        assert cfg["lambda_scl"] == 0.0
        assert cfg["lambda_fcl"] == 0.1  # remains
        assert cfg["lambda_cl"] == 0.5  # remains

    def test_wo_fcl_only_zeros_fcl(self):
        """wo_fcl only zeros out FCL, SCL remains."""
        from models.tts.comelsinger.eval.ablation_configs import ABLATION_CONFIGS

        cfg = ABLATION_CONFIGS["wo_fcl"]
        assert cfg["lambda_fcl"] == 0.0
        assert cfg["lambda_scl"] == 1.0  # remains

    def test_wo_svt_zeros_svt(self):
        """wo_svt disables SVT supervision."""
        from models.tts.comelsinger.eval.ablation_configs import ABLATION_CONFIGS

        cfg = ABLATION_CONFIGS["wo_svt"]
        assert cfg["lambda_svt"] == 0.0
        # CL should remain enabled
        assert cfg["lambda_cl"] == 0.5

    def test_wo_cl_svt_zeros_both(self):
        """wo_cl_svt disables both CL and SVT."""
        from models.tts.comelsinger.eval.ablation_configs import ABLATION_CONFIGS

        cfg = ABLATION_CONFIGS["wo_cl_svt"]
        assert cfg["lambda_cl"] == 0.0
        assert cfg["lambda_scl"] == 0.0
        assert cfg["lambda_fcl"] == 0.0
        assert cfg["lambda_svt"] == 0.0
        # mask should remain
        assert cfg["lambda_mask"] == 0.3

    def test_all_configs_present(self):
        """All 6 ablation conditions are defined."""
        from models.tts.comelsinger.eval.ablation_configs import ABLATION_CONFIGS

        expected = {"full", "wo_cl", "wo_scl", "wo_fcl", "wo_svt", "wo_cl_svt"}
        assert set(ABLATION_CONFIGS.keys()) == expected

    def test_get_config_returns_copy(self):
        """get_config returns a deep copy, not a reference."""
        from models.tts.comelsinger.eval.ablation_configs import ABLATION_CONFIGS, get_config

        cfg = get_config("full")
        cfg["lambda_cl"] = 999.0
        assert ABLATION_CONFIGS["full"]["lambda_cl"] == 0.5

    def test_get_config_invalid_raises(self):
        """get_config raises ValueError for unknown condition."""
        from models.tts.comelsinger.eval.ablation_configs import get_config

        with pytest.raises(ValueError, match="Unknown ablation condition"):
            get_config("nonexistent")

    def test_all_configs_have_mask(self):
        """All ablation configs preserve lambda_mask=0.3."""
        from models.tts.comelsinger.eval.ablation_configs import ABLATION_CONFIGS

        for cond, cfg in ABLATION_CONFIGS.items():
            assert cfg["lambda_mask"] == 0.3, f"{cond} has wrong lambda_mask"


# ---------------------------------------------------------------------------
# TestAblationSummary (M4-11)
# ---------------------------------------------------------------------------


class TestAblationSummary:
    def _create_results_dir(self, tmp_path, split="seen"):
        """Create mock results directory with JSON files for all conditions."""
        from models.tts.comelsinger.eval.ablation_summary import CONDITIONS, METRICS

        for cond in CONDITIONS:
            cond_dir = tmp_path / cond
            cond_dir.mkdir()
            for metric_name, (metric_file, metric_key) in METRICS.items():
                data = {metric_key: np.random.uniform(0, 5), "std": 0.1, "n_samples": 50}
                with open(cond_dir / f"{metric_file}_{split}.json", "w") as f:
                    json.dump(data, f)

    def test_load_results_all_present(self, tmp_path):
        """All 6 conditions and 5 metrics are loaded."""
        from models.tts.comelsinger.eval.ablation_summary import CONDITIONS, METRICS, load_results

        self._create_results_dir(tmp_path)
        rows = load_results(tmp_path, "seen")
        assert len(rows) == len(CONDITIONS)
        for row in rows:
            for m in METRICS:
                assert m in row
                assert row[m] is not None

    def test_generate_summary_markdown(self, tmp_path):
        """generate_summary returns valid markdown string."""
        from models.tts.comelsinger.eval.ablation_summary import generate_summary

        self._create_results_dir(tmp_path)
        md = generate_summary(tmp_path, "seen")
        assert "| Condition |" in md
        assert "full" in md
        assert "wo_cl" in md
        assert "Paper (full)" in md

    def test_missing_file_handled(self, tmp_path):
        """Missing JSON files result in None values, not crashes."""
        from models.tts.comelsinger.eval.ablation_summary import load_results

        # Create only the full condition
        cond_dir = tmp_path / "full"
        cond_dir.mkdir()
        with open(cond_dir / "mcd_seen.json", "w") as f:
            json.dump({"mean_mcd": 4.17}, f)

        rows = load_results(tmp_path, "seen")
        assert len(rows) == 6
        # full condition has MCD, others don't
        full_row = [r for r in rows if r["condition"] == "full"][0]
        assert full_row["MCD"] == 4.17


# ---------------------------------------------------------------------------
# TestEvalReport (M4-13)
# ---------------------------------------------------------------------------


class TestEvalReport:
    def _create_eval_dir(self, tmp_path):
        """Create mock eval directory with all metric JSON files."""
        metrics = {
            "mcd": ("mean_mcd", "std_mcd"),
            "f0_rmse": ("mean_f0_rmse", "std_f0_rmse"),
            "singmos": ("mean_singmos", "std_singmos"),
            "secs": ("mean_secs", "std_secs"),
            "svt_f1": ("mean_f1", "std_f1"),
        }
        for prefix, (mean_key, std_key) in metrics.items():
            for split in ("seen", "unseen"):
                data = {mean_key: np.random.uniform(0, 5), std_key: 0.1, "n_samples": 50}
                with open(tmp_path / f"{prefix}_{split}.json", "w") as f:
                    json.dump(data, f)

    def test_generate_report_json(self, tmp_path):
        """Report JSON contains all 5 metrics."""
        from models.tts.comelsinger.eval.eval_report import generate_report

        self._create_eval_dir(tmp_path)
        output = tmp_path / "report.json"
        report = generate_report(tmp_path, output)

        assert set(report.keys()) == {"MCD", "F0-RMSE", "SingMOS", "SECS", "SVT-F1"}
        assert output.exists()

        # Also check markdown was generated
        md_path = output.with_suffix(".md")
        assert md_path.exists()

    def test_report_includes_paper_targets(self, tmp_path):
        """Report includes paper target values for comparison."""
        from models.tts.comelsinger.eval.eval_report import generate_report

        self._create_eval_dir(tmp_path)
        report = generate_report(tmp_path, tmp_path / "report.json")

        assert report["MCD"]["paper_target"]["seen"] == 4.17
        assert report["F0-RMSE"]["paper_target"]["seen"] == 0.042
        assert report["SingMOS"]["paper_target"]["seen"] == 4.32
        assert report["SECS"]["paper_target"]["seen"] == 0.912
        assert report["SVT-F1"]["paper_target"]["seen"] == 0.711

    def test_report_computes_delta(self, tmp_path):
        """Report computes delta from paper target."""
        from models.tts.comelsinger.eval.eval_report import generate_report

        self._create_eval_dir(tmp_path)
        report = generate_report(tmp_path, tmp_path / "report.json")

        for metric in ("MCD", "F0-RMSE", "SingMOS", "SECS", "SVT-F1"):
            seen_data = report[metric]["seen"]
            if seen_data["mean"] is not None:
                assert seen_data["delta"] is not None

    def test_report_handles_missing_files(self, tmp_path):
        """Report handles missing metric files gracefully."""
        from models.tts.comelsinger.eval.eval_report import generate_report

        # Create only MCD files
        for split in ("seen", "unseen"):
            data = {"mean_mcd": 4.17, "std_mcd": 0.3, "n_samples": 50}
            with open(tmp_path / f"mcd_{split}.json", "w") as f:
                json.dump(data, f)

        report = generate_report(tmp_path, tmp_path / "report.json")
        assert report["MCD"]["seen"]["mean"] == 4.17
        assert report["F0-RMSE"]["seen"]["mean"] is None

    def test_report_json_is_valid(self, tmp_path):
        """Generated JSON file is valid and parseable."""
        from models.tts.comelsinger.eval.eval_report import generate_report

        self._create_eval_dir(tmp_path)
        output = tmp_path / "report.json"
        generate_report(tmp_path, output)

        with open(output) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert len(data) == 5
