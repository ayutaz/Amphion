"""Tests for S2A training script (M3-08 to M3-22)."""
from __future__ import annotations

from typing import Dict

import pytest
import torch


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _dummy_batch(B: int = 8, T: int = 50) -> Dict[str, torch.Tensor]:
    """Create a dummy batch matching CoMelSingerDataset/collate output."""
    # Repeat speaker pattern to handle any batch size
    spk_pattern = [0, 0, 1, 1, 2, 2, 3, 3]
    speaker_ids = (spk_pattern * ((B // len(spk_pattern)) + 1))[:B]
    # Note durations must sum to T for correct SVT loss computation
    n_notes = 5
    dur_per_note = T // n_notes
    remainder = T - dur_per_note * n_notes
    durations = [dur_per_note] * n_notes
    durations[-1] += remainder  # absorb remainder into last note
    return {
        "acoustic_tokens": torch.randint(0, 1024, (B, 12, T)),
        "semantic_tokens": torch.randint(0, 1024, (B, T)),
        "pitch_tokens": torch.randint(1, 129, (B, T)),  # 1-128 voiced
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "speaker_id": torch.tensor(speaker_ids),
        "note_durations": [torch.tensor(durations) for _ in range(B)],
        "note_pitches": [torch.tensor([60, 62, 64, 65, 67]) for _ in range(B)],
    }


# ===========================================================================
# TestPitchPerturbation (M3-10)
# ===========================================================================

class TestPitchPerturbation:
    """Test pitch_perturbation per-sample independence, shape, and value range."""

    def test_shape_preserved(self):
        from models.tts.comelsinger.train_s2a import pitch_perturbation

        pitch = torch.randint(0, 129, (4, 100))
        perturbed = pitch_perturbation(pitch)
        assert perturbed.shape == pitch.shape

    def test_value_range(self):
        from models.tts.comelsinger.train_s2a import pitch_perturbation

        pitch = torch.randint(0, 129, (8, 50))
        perturbed = pitch_perturbation(pitch, zero_prob=0.0, max_shift=12)
        assert perturbed.min() >= 0
        assert perturbed.max() <= 128

    def test_zero_prob_one_all_zeros(self):
        from models.tts.comelsinger.train_s2a import pitch_perturbation

        pitch = torch.ones(4, 20, dtype=torch.long) * 60
        perturbed = pitch_perturbation(pitch, zero_prob=1.0)
        assert (perturbed == 0).all()

    def test_zero_prob_zero_no_zeros(self):
        from models.tts.comelsinger.train_s2a import pitch_perturbation

        pitch = torch.ones(4, 20, dtype=torch.long) * 64
        perturbed = pitch_perturbation(pitch, zero_prob=0.0, max_shift=0)
        # With max_shift=0 and zero_prob=0, output should equal input
        assert torch.equal(perturbed, pitch)

    def test_sample_independence(self):
        """With enough samples, both zeroed and shifted rows should appear."""
        from models.tts.comelsinger.train_s2a import pitch_perturbation

        torch.manual_seed(42)
        pitch = torch.ones(100, 10, dtype=torch.long) * 64
        perturbed = pitch_perturbation(pitch, zero_prob=0.5)
        zero_rows = (perturbed == 0).all(dim=1).sum().item()
        # Statistical: expect ~50 zero rows, accept 20-80
        assert 20 <= zero_rows <= 80, f"zero_rows={zero_rows}, expected ~50"

    def test_original_not_modified(self):
        from models.tts.comelsinger.train_s2a import pitch_perturbation

        pitch = torch.ones(2, 10, dtype=torch.long) * 60
        original = pitch.clone()
        _ = pitch_perturbation(pitch, zero_prob=1.0)
        assert torch.equal(pitch, original), "Original tensor was modified in-place"

    def test_clamp_upper_boundary(self):
        from models.tts.comelsinger.train_s2a import pitch_perturbation

        pitch = torch.full((4, 10), 128, dtype=torch.long)
        perturbed = pitch_perturbation(pitch, zero_prob=0.0, max_shift=6)
        assert perturbed.max() <= 128

    def test_clamp_lower_boundary(self):
        from models.tts.comelsinger.train_s2a import pitch_perturbation

        pitch = torch.full((4, 10), 1, dtype=torch.long)
        perturbed = pitch_perturbation(pitch, zero_prob=0.0, max_shift=6)
        assert perturbed.min() >= 0


# ===========================================================================
# TestBatchSplit (M3-09)
# ===========================================================================

class TestBatchSplit:
    """Test batch splitting into SCL and FCL subsets."""

    def test_split_shapes(self):
        from models.tts.comelsinger.train_s2a import split_batch

        batch = _dummy_batch(B=32, T=50)
        s_a_s, s_a_f = split_batch(batch, k_s=8)
        assert s_a_s["acoustic_tokens"].shape[0] == 8
        assert s_a_f["acoustic_tokens"].shape[0] == 24
        assert s_a_s["semantic_tokens"].shape == (8, 50)
        assert s_a_f["semantic_tokens"].shape == (24, 50)

    def test_split_covers_all_samples(self):
        from models.tts.comelsinger.train_s2a import split_batch

        batch = _dummy_batch(B=16, T=30)
        s_a_s, s_a_f = split_batch(batch, k_s=4)
        total = s_a_s["pitch_tokens"].shape[0] + s_a_f["pitch_tokens"].shape[0]
        assert total == 16

    def test_split_preserves_values(self):
        from models.tts.comelsinger.train_s2a import split_batch

        batch = _dummy_batch(B=8, T=20)
        s_a_s, s_a_f = split_batch(batch, k_s=3)
        assert torch.equal(s_a_s["pitch_tokens"], batch["pitch_tokens"][:3])
        assert torch.equal(s_a_f["pitch_tokens"], batch["pitch_tokens"][3:])

    def test_split_list_fields(self):
        from models.tts.comelsinger.train_s2a import split_batch

        batch = _dummy_batch(B=8, T=20)
        s_a_s, s_a_f = split_batch(batch, k_s=3)
        assert len(s_a_s["note_durations"]) == 3
        assert len(s_a_f["note_durations"]) == 5


# ===========================================================================
# TestInverseSqrtScheduler (M3-08)
# ===========================================================================

class TestInverseSqrtScheduler:
    """Test inverse square-root LR schedule with warmup."""

    def test_warmup_phase(self):
        from models.tts.comelsinger.train_s2a import inverse_sqrt_schedule

        # During warmup: lr_mult = (step+1) / warmup_steps
        assert inverse_sqrt_schedule(0, warmup_steps=100) == pytest.approx(1 / 100)
        assert inverse_sqrt_schedule(49, warmup_steps=100) == pytest.approx(50 / 100)
        assert inverse_sqrt_schedule(99, warmup_steps=100) == pytest.approx(100 / 100)

    def test_post_warmup_decay(self):
        from models.tts.comelsinger.train_s2a import inverse_sqrt_schedule

        warmup = 1000
        # At warmup boundary: sqrt(1000)/sqrt(1001) ~ 1.0
        val_at_warmup = inverse_sqrt_schedule(1000, warmup_steps=warmup)
        val_at_2000 = inverse_sqrt_schedule(2000, warmup_steps=warmup)
        val_at_4000 = inverse_sqrt_schedule(4000, warmup_steps=warmup)
        # Should be monotonically decreasing
        assert val_at_warmup > val_at_2000 > val_at_4000

    def test_lr_decreases_over_time(self):
        from models.tts.comelsinger.train_s2a import inverse_sqrt_schedule

        warmup = 100
        values = [inverse_sqrt_schedule(s, warmup) for s in range(200, 1000, 100)]
        for i in range(1, len(values)):
            assert values[i] < values[i - 1], "LR should decrease after warmup"

    def test_pytorch_scheduler_integration(self):
        from models.tts.comelsinger.train_s2a import inverse_sqrt_schedule

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: inverse_sqrt_schedule(step, 100),
        )
        # Step 0: warmup
        lr0 = scheduler.get_last_lr()[0]
        assert lr0 > 0
        # After several steps, LR should still be positive
        for _ in range(200):
            scheduler.step()
        lr200 = scheduler.get_last_lr()[0]
        assert lr200 > 0


# ===========================================================================
# TestSoftLabelMatrix (M3-14)
# ===========================================================================

class TestSoftLabelMatrix:
    """Test soft label matrix construction for FCL."""

    def test_shape(self):
        from models.tts.comelsinger.train_s2a import build_soft_label_matrix

        pitch = torch.tensor([[60, 60, 62, 0, 62]])
        Y = build_soft_label_matrix(pitch)
        assert Y.shape == (1, 5, 5)

    def test_unvoiced_excluded(self):
        from models.tts.comelsinger.train_s2a import build_soft_label_matrix

        pitch = torch.tensor([[0, 60, 60]])
        Y = build_soft_label_matrix(pitch)
        # Row/col 0 (unvoiced) should be all zeros
        assert Y[0, 0, :].sum() == 0
        assert Y[0, :, 0].sum() == 0

    def test_same_pitch_positive(self):
        from models.tts.comelsinger.train_s2a import build_soft_label_matrix

        pitch = torch.tensor([[60, 60, 62]])
        Y = build_soft_label_matrix(pitch)
        assert Y[0, 0, 1] == 1.0  # same pitch
        assert Y[0, 1, 0] == 1.0  # symmetric
        assert Y[0, 0, 2] == 0.0  # different pitch

    def test_batched(self):
        from models.tts.comelsinger.train_s2a import build_soft_label_matrix

        pitch = torch.tensor([[60, 60, 62], [0, 64, 64]])
        Y = build_soft_label_matrix(pitch)
        assert Y.shape == (2, 3, 3)


# ===========================================================================
# TestPromptGen (M3-11)
# ===========================================================================

class TestPromptGen:
    """Test prompt generation from same-speaker samples."""

    def test_basic_prompt_gen(self):
        from models.tts.comelsinger.train_s2a import prompt_gen

        batch = _dummy_batch(B=8, T=50)
        prompt = prompt_gen(batch, max_len=30)
        # acoustic_tokens is (B, 12, T) -> truncated on last dim
        assert prompt["acoustic_tokens"].shape == (8, 12, 30)
        # semantic_tokens is (B, T) -> truncated on dim 1
        assert prompt["semantic_tokens"].shape == (8, 30)

    def test_prompt_different_index(self):
        from models.tts.comelsinger.train_s2a import prompt_gen

        batch = _dummy_batch(B=8, T=50)
        # Run multiple times to verify randomness works
        for _ in range(5):
            prompt = prompt_gen(batch, max_len=50)
            # Speaker IDs in prompt should exist in original
            for sid in prompt["speaker_id"]:
                assert sid.item() in batch["speaker_id"].tolist()

    def test_single_speaker_fallback(self):
        from models.tts.comelsinger.train_s2a import prompt_gen

        batch = {
            "acoustic_tokens": torch.zeros(4, 12, 100, dtype=torch.long),
            "semantic_tokens": torch.zeros(4, 100, dtype=torch.long),
            "pitch_tokens": torch.zeros(4, 100, dtype=torch.long),
            "attention_mask": torch.ones(4, 100, dtype=torch.long),
            "speaker_id": torch.zeros(4, dtype=torch.long),  # all same speaker
        }
        # Should not raise even with single speaker
        prompt = prompt_gen(batch, max_len=50)
        assert prompt["acoustic_tokens"].shape[0] == 4


# ===========================================================================
# TestAlgorithm1Step (M3-21 smoke test)
# ===========================================================================

class TestAlgorithm1Step:
    """Test that one Algorithm 1 step completes without errors (smoke test)."""

    @pytest.fixture(scope="class")
    def models_and_cfg(self):
        """Build models once for all tests in this class."""
        from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
        from models.tts.comelsinger.svt_module import SVTModule

        cfg = {
            "model": {"pitch_vocab_size": 129, "temperature": 0.07},
            "loss": {
                "lambda_cl": 0.5, "lambda_scl": 1.0, "lambda_fcl": 0.1,
                "lambda_svt": 0.5, "lambda_mask": 0.3,
                "lambda_seg": 3.0, "lambda_dur": 5.0,
                "delta": 0.5, "tau": 0.07,
            },
            "training": {"k_s": 2, "batch_size": 4},
        }
        device = torch.device("cpu")
        s2a = CoMelSinger_S2A()
        svt = SVTModule()
        svt.freeze()
        return s2a, svt, cfg, device

    def test_one_step_forward(self, models_and_cfg):
        """1 step of Algorithm 1 should produce finite losses."""
        from models.tts.comelsinger.train_s2a import algorithm1_step

        s2a, svt, cfg, device = models_and_cfg
        s2a.train()

        batch = _dummy_batch(B=4, T=30)
        l_total, loss_dict = algorithm1_step(s2a, svt, batch, cfg, device)

        assert torch.isfinite(l_total), f"l_total is not finite: {l_total}"
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                assert torch.isfinite(v), f"{k} is not finite: {v}"

    def test_one_step_backward(self, models_and_cfg):
        """backward() should complete without error."""
        from models.tts.comelsinger.train_s2a import algorithm1_step

        s2a, svt, cfg, device = models_and_cfg
        s2a.train()
        s2a.zero_grad()

        batch = _dummy_batch(B=4, T=30)
        l_total, _ = algorithm1_step(s2a, svt, batch, cfg, device)
        l_total.backward()

        # At least some parameters should have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in s2a.parameters() if p.requires_grad
        )
        assert has_grad, "No gradients flowed to any trainable parameter"

    def test_loss_dict_keys(self, models_and_cfg):
        """Loss dict should contain all expected keys."""
        from models.tts.comelsinger.train_s2a import algorithm1_step

        s2a, svt, cfg, device = models_and_cfg
        s2a.train()

        batch = _dummy_batch(B=4, T=30)
        _, loss_dict = algorithm1_step(s2a, svt, batch, cfg, device)

        required_keys = {"l_scl", "l_fcl", "l_cl", "l_svt", "l_mask", "l_total"}
        for key in required_keys:
            assert key in loss_dict, f"Missing key: {key}"


# ===========================================================================
# TestLambdaCrosscheck (M3-22)
# ===========================================================================

class TestLambdaCrosscheck:
    """Verify config file values match code defaults."""

    def test_code_defaults_match_expected(self):
        """EXPECTED_LAMBDAS in train_s2a.py should match the requirements."""
        from models.tts.comelsinger.train_s2a import EXPECTED_LAMBDAS

        assert EXPECTED_LAMBDAS["lambda_cl"] == 0.5
        assert EXPECTED_LAMBDAS["lambda_scl"] == 1.0
        assert EXPECTED_LAMBDAS["lambda_fcl"] == 0.1
        assert EXPECTED_LAMBDAS["lambda_svt"] == 0.5
        assert EXPECTED_LAMBDAS["lambda_mask"] == 0.3
        assert EXPECTED_LAMBDAS["lambda_seg"] == 3.0
        assert EXPECTED_LAMBDAS["lambda_dur"] == 5.0
        assert EXPECTED_LAMBDAS["tau"] == 0.07

    def test_config_file_matches_expected(self):
        """s2a_train.yaml loss values should match EXPECTED_LAMBDAS."""
        import os
        import yaml
        from models.tts.comelsinger.train_s2a import EXPECTED_LAMBDAS

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..",
            "configs", "comelsinger", "s2a_train.yaml",
        )
        if not os.path.exists(config_path):
            pytest.skip("s2a_train.yaml not found")

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        loss_cfg = cfg.get("loss", {})
        for key, expected in EXPECTED_LAMBDAS.items():
            actual = loss_cfg.get(key)
            if actual is not None:
                assert actual == expected, (
                    f"Config mismatch: loss.{key} = {actual}, expected {expected}"
                )

    def test_model_defaults_match(self):
        """CoMelSinger_S2A default hyperparameters should match config."""
        from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
        import inspect

        sig = inspect.signature(CoMelSinger_S2A.__init__)
        params = sig.parameters
        assert params["lambda_cl"].default == 0.5
        assert params["lambda_scl"].default == 1.0
        assert params["lambda_fcl"].default == 0.1
        assert params["lambda_svt"].default == 0.5
        assert params["lambda_mask"].default == 0.3
        assert params["temperature"].default == 0.07

    def test_crosscheck_function(self):
        """crosscheck_config should pass with valid config."""
        from models.tts.comelsinger.train_s2a import crosscheck_config

        valid_cfg = {
            "loss": {
                "lambda_cl": 0.5,
                "lambda_scl": 1.0,
                "lambda_fcl": 0.1,
                "lambda_svt": 0.5,
                "lambda_mask": 0.3,
                "lambda_seg": 3.0,
                "lambda_dur": 5.0,
                "tau": 0.07,
            }
        }
        # Should not raise
        crosscheck_config(valid_cfg)

    def test_crosscheck_function_fails_on_mismatch(self):
        """crosscheck_config should raise AssertionError on mismatch."""
        from models.tts.comelsinger.train_s2a import crosscheck_config

        bad_cfg = {
            "loss": {
                "lambda_cl": 0.5,
                "lambda_scl": 0.5,  # wrong: should be 1.0
                "lambda_fcl": 0.1,
            }
        }
        with pytest.raises(AssertionError, match="lambda_scl"):
            crosscheck_config(bad_cfg)


# ===========================================================================
# TestLogging (M3-19)
# ===========================================================================

class TestLogging:
    """Test TensorBoard logging function."""

    def test_log_metrics_main_process(self):
        from models.tts.comelsinger.train_s2a import log_metrics

        class MockWriter:
            def __init__(self):
                self.logged = {}

            def add_scalar(self, tag, value, step):
                self.logged[tag] = (value, step)

        writer = MockWriter()
        loss_dict = {
            "l_scl": torch.tensor(1.0), "l_fcl": torch.tensor(0.5),
            "l_cl": torch.tensor(1.05), "l_svt": torch.tensor(0.8),
            "l_mask": torch.tensor(2.0), "l_total": torch.tensor(1.6),
        }
        log_metrics(writer, loss_dict, lr=1e-5, grad_norm=0.9,
                     global_step=100, is_main_process=True)
        assert "train/l_total" in writer.logged
        assert "train/lr" in writer.logged
        assert "train/grad_norm" in writer.logged

    def test_log_metrics_non_main_skips(self):
        from models.tts.comelsinger.train_s2a import log_metrics

        class MockWriter:
            def __init__(self):
                self.call_count = 0

            def add_scalar(self, *args, **kwargs):
                self.call_count += 1

        writer = MockWriter()
        log_metrics(writer, {"l_total": 1.0}, lr=1e-5, grad_norm=0.0,
                     global_step=1, is_main_process=False)
        assert writer.call_count == 0, "Non-main process should not log"
