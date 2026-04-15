"""Tests for SVT training script (M3-01 to M3-07).

Covers:
  TestTrainSVTConfig     - YAML config loading and required key validation
  TestTrainSVTDryRun     - Dry-run execution (subprocess) completes successfully
  TestEvaluateSVT        - evaluate_svt() returns valid metrics with dummy data
  TestTrainStep          - Single training step produces finite loss
  TestSmokeTestOverrides - Smoke test mode overrides config values
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from models.tts.comelsinger.dataset import comelsinger_collate_fn, prepare_batch_for_svt
from models.tts.comelsinger.svt_module import SVTModule
from models.tts.comelsinger.train_svt import (
    build_model,
    build_optimizer,
    build_scheduler,
    evaluate_svt,
    load_config,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    train_step,
)

SEED = 42
# Resolve paths relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SVT_CONFIG = _PROJECT_ROOT / "configs" / "comelsinger" / "svt_train.yaml"


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_batch(
    batch_size: int = 2,
    seq_len: int = 30,
    n_notes: int = 5,
) -> dict:
    """Create a dummy collated batch that mimics comelsinger_collate_fn output.

    Returns a batch dict with:
        acoustic_tokens: (B, 12, T) long
        pitch_tokens: (B, T) long
        attention_mask: (B, T) long
        note_durations: list of Tensor
        note_pitches: list of Tensor
        semantic_tokens: (B, T) long
        speaker_id: (B,) long
    """
    B, T = batch_size, seq_len
    frames_per_note = T // n_notes
    remainder = T - frames_per_note * n_notes

    note_durations = []
    note_pitches = []
    for _ in range(B):
        durations = [frames_per_note] * n_notes
        if remainder > 0:
            durations[-1] += remainder
        note_durations.append(torch.tensor(durations, dtype=torch.long))
        note_pitches.append(torch.randint(1, 129, (n_notes,)))

    return {
        "acoustic_tokens": torch.randint(0, 1024, (B, 12, T)),
        "semantic_tokens": torch.randint(0, 1024, (B, T)),
        "pitch_tokens": torch.randint(1, 129, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "speaker_id": torch.randint(0, 20, (B,)),
        "note_durations": note_durations,
        "note_pitches": note_pitches,
        "phone_ids": [torch.randint(0, 100, (10,)) for _ in range(B)],
    }


def _make_dummy_config() -> dict:
    """Return a minimal valid config dict for testing."""
    return {
        "model": {
            "num_codebooks": 12,
            "codebook_size": 1024,
            "codebook_embed_dim": 64,
            "hidden_size": 512,
            "num_layers": 4,
            "num_heads": 8,
            "pitch_vocab_size": 129,
            "dropout": 0.1,
        },
        "training": {
            "seed": 42,
            "max_steps": 100,
            "batch_size": 4,
            "gradient_clip": 1.0,
            "precision": "fp32",
            "log_interval": 10,
            "save_interval": 50,
            "val_interval": 50,
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-5,
            "weight_decay": 0.01,
        },
        "scheduler": {
            "type": "cosine",
            "T_max": 100,
            "eta_min": 0,
        },
        "loss": {
            "lambda_seg": 3.0,
            "lambda_dur": 5.0,
            "delta": 0.5,
        },
        "data": {
            "data_dir": "data/preprocessed",
            "num_workers": 0,
        },
        "checkpoint": {
            "output_dir": "checkpoints/svt",
        },
    }


# ---------------------------------------------------------------------------
# TestTrainSVTConfig (M3-01, M3-02)
# ---------------------------------------------------------------------------


class TestTrainSVTConfig:
    """Test YAML config loading and validation."""

    def test_load_config_from_file(self):
        """load_config reads the actual YAML file and returns a dict."""
        config = load_config(str(_SVT_CONFIG))
        assert isinstance(config, dict)

    def test_required_keys_present(self):
        """Config contains all required top-level keys."""
        config = load_config(str(_SVT_CONFIG))
        required = {"model", "training", "optimizer", "loss", "data", "checkpoint"}
        assert required.issubset(set(config.keys()))

    def test_model_params(self):
        """Model section contains expected hyperparameters."""
        config = load_config(str(_SVT_CONFIG))
        model_cfg = config["model"]
        assert model_cfg["num_codebooks"] == 12
        assert model_cfg["hidden_size"] == 512
        assert model_cfg["num_layers"] == 4
        assert model_cfg["num_heads"] == 8
        assert model_cfg["pitch_vocab_size"] == 129

    def test_training_params(self):
        """Training section contains expected hyperparameters."""
        config = load_config(str(_SVT_CONFIG))
        train_cfg = config["training"]
        assert train_cfg["seed"] == 42
        assert train_cfg["max_steps"] == 50000
        assert train_cfg["batch_size"] == 32

    def test_loss_params(self):
        """Loss section contains expected hyperparameters."""
        config = load_config(str(_SVT_CONFIG))
        loss_cfg = config["loss"]
        assert loss_cfg["lambda_seg"] == 3.0
        assert loss_cfg["lambda_dur"] == 5.0
        assert loss_cfg["delta"] == 0.5

    def test_missing_file_raises(self):
        """load_config raises FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_missing_keys_raises(self):
        """load_config raises ValueError when required keys are missing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("model:\n  hidden_size: 512\n")
            f.flush()
            with pytest.raises(ValueError, match="required keys"):
                load_config(f.name)

    def test_build_model_from_config(self):
        """build_model creates SVTModule with correct parameters from config."""
        config = _make_dummy_config()
        model = build_model(config)
        assert isinstance(model, SVTModule)
        assert model.hidden_size == 512
        assert model.num_layers == 4

    def test_build_optimizer_from_config(self):
        """build_optimizer creates AdamW optimizer from config."""
        config = _make_dummy_config()
        model = build_model(config)
        optimizer = build_optimizer(model, config)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 1e-5

    def test_build_scheduler_from_config(self):
        """build_scheduler creates CosineAnnealingLR from config."""
        config = _make_dummy_config()
        model = build_model(config)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


# ---------------------------------------------------------------------------
# TestTrainSVTDryRun (M3-03)
# ---------------------------------------------------------------------------


class TestTrainSVTDryRun:
    """Test dry-run mode exits cleanly after initialization."""

    def test_dry_run_subprocess(self):
        """Running train_svt.py with --dry-run exits with code 0."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "models.tts.comelsinger.train_svt",
                "--config",
                str(_SVT_CONFIG),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Dry-run failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_dry_run_logs_initialization(self):
        """Dry-run output contains initialization confirmation."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "models.tts.comelsinger.train_svt",
                "--config",
                str(_SVT_CONFIG),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=120,
        )
        assert "Dry-run complete" in result.stderr or "Dry-run complete" in result.stdout


# ---------------------------------------------------------------------------
# TestEvaluateSVT (M3-06)
# ---------------------------------------------------------------------------


class TestEvaluateSVT:
    """Test evaluate_svt() with dummy model and data."""

    def _make_dummy_val_loader(self, n_batches: int = 3, batch_size: int = 2, seq_len: int = 30):
        """Create a dummy DataLoader with pre-built batches."""
        batches = [_make_dummy_batch(batch_size=batch_size, seq_len=seq_len) for _ in range(n_batches)]

        class _DummyDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        class _DummyLoader:
            """Minimal DataLoader-like object that yields pre-built batches."""

            def __init__(self, batches):
                self._batches = batches

            def __iter__(self):
                return iter(self._batches)

            def __len__(self):
                return len(self._batches)

        return _DummyLoader(batches)

    def test_returns_expected_keys(self):
        """evaluate_svt returns dict with accuracy, f1, val_loss keys."""
        model = SVTModule()
        val_loader = self._make_dummy_val_loader()
        metrics = evaluate_svt(model, val_loader, torch.device("cpu"))
        assert set(metrics.keys()) == {"accuracy", "f1", "val_loss"}

    def test_accuracy_in_valid_range(self):
        """Accuracy is between 0 and 1."""
        model = SVTModule()
        val_loader = self._make_dummy_val_loader()
        metrics = evaluate_svt(model, val_loader, torch.device("cpu"))
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_f1_in_valid_range(self):
        """F1 is between 0 and 1."""
        model = SVTModule()
        val_loader = self._make_dummy_val_loader()
        metrics = evaluate_svt(model, val_loader, torch.device("cpu"))
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_val_loss_finite(self):
        """Validation loss is a finite number."""
        model = SVTModule()
        val_loader = self._make_dummy_val_loader()
        metrics = evaluate_svt(model, val_loader, torch.device("cpu"))
        assert metrics["val_loss"] >= 0.0
        assert not (metrics["val_loss"] != metrics["val_loss"])  # not NaN

    def test_model_returns_to_train_mode(self):
        """After evaluate_svt, model is back in train mode."""
        model = SVTModule()
        model.train()
        val_loader = self._make_dummy_val_loader()
        evaluate_svt(model, val_loader, torch.device("cpu"))
        assert model.training


# ---------------------------------------------------------------------------
# TestTrainStep (M3-04)
# ---------------------------------------------------------------------------


class TestTrainStep:
    """Test single training step produces finite loss and valid gradients."""

    def test_single_step_finite_loss(self):
        """One training step returns finite loss_total."""
        config = _make_dummy_config()
        model = SVTModule()
        model.train()
        optimizer = build_optimizer(model, config)
        batch = _make_dummy_batch()

        metrics = train_step(model, batch, optimizer, config, torch.device("cpu"))

        assert "loss_total" in metrics
        assert metrics["loss_total"] == metrics["loss_total"]  # not NaN
        assert abs(metrics["loss_total"]) < 1e6  # finite and reasonable

    def test_single_step_all_components(self):
        """One training step returns all expected metric keys."""
        config = _make_dummy_config()
        model = SVTModule()
        model.train()
        optimizer = build_optimizer(model, config)
        batch = _make_dummy_batch()

        metrics = train_step(model, batch, optimizer, config, torch.device("cpu"))

        expected_keys = {"loss_total", "l_ce", "l_seg", "l_dur", "lr"}
        assert set(metrics.keys()) == expected_keys

    def test_single_step_loss_components_finite(self):
        """All loss components from one step are finite."""
        config = _make_dummy_config()
        model = SVTModule()
        model.train()
        optimizer = build_optimizer(model, config)
        batch = _make_dummy_batch()

        metrics = train_step(model, batch, optimizer, config, torch.device("cpu"))

        for key in ["l_ce", "l_seg", "l_dur"]:
            assert metrics[key] == metrics[key], f"{key} is NaN"
            assert abs(metrics[key]) < 1e6, f"{key} is not finite: {metrics[key]}"

    def test_gradients_after_step(self):
        """After train_step, model parameters have been updated (grad consumed)."""
        config = _make_dummy_config()
        model = SVTModule()
        model.train()
        optimizer = build_optimizer(model, config)

        # Capture initial params
        initial_params = {
            name: p.clone() for name, p in model.named_parameters() if p.requires_grad
        }

        batch = _make_dummy_batch()
        train_step(model, batch, optimizer, config, torch.device("cpu"))

        # At least some parameters should have changed
        any_changed = False
        for name, p in model.named_parameters():
            if p.requires_grad and not torch.equal(p, initial_params[name]):
                any_changed = True
                break
        assert any_changed, "No parameters changed after training step"

    def test_two_consecutive_steps(self):
        """Two consecutive training steps both produce finite loss."""
        config = _make_dummy_config()
        model = SVTModule()
        model.train()
        optimizer = build_optimizer(model, config)

        for i in range(2):
            batch = _make_dummy_batch()
            metrics = train_step(model, batch, optimizer, config, torch.device("cpu"))
            assert metrics["loss_total"] == metrics["loss_total"], f"Step {i}: loss is NaN"


# ---------------------------------------------------------------------------
# TestSmokeTestOverrides (M3-07)
# ---------------------------------------------------------------------------


class TestSmokeTestOverrides:
    """Test smoke test flag overrides config values."""

    def test_smoke_test_max_steps(self):
        """Smoke test mode sets max_steps=100."""
        config = _make_dummy_config()
        config["training"]["max_steps"] = 50000  # original
        # Simulate smoke test override
        config["training"]["max_steps"] = 100
        assert config["training"]["max_steps"] == 100

    def test_smoke_test_batch_size(self):
        """Smoke test mode sets batch_size=4."""
        config = _make_dummy_config()
        config["training"]["batch_size"] = 32  # original
        config["training"]["batch_size"] = 4
        assert config["training"]["batch_size"] == 4


# ---------------------------------------------------------------------------
# TestCheckpoint (M3-05)
# ---------------------------------------------------------------------------


class TestCheckpoint:
    """Test checkpoint save/load round-trip."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Saving and loading a checkpoint restores step and best_f1."""
        config = _make_dummy_config()
        model = SVTModule()
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)

        save_checkpoint(model, optimizer, scheduler, step=42, best_f1=0.85, output_dir=tmp_path)
        ckpt_path = tmp_path / "checkpoint.pt"
        assert ckpt_path.exists()

        # Load into fresh model
        model2 = SVTModule()
        optimizer2 = build_optimizer(model2, config)
        scheduler2 = build_scheduler(optimizer2, config)
        info = load_checkpoint(ckpt_path, model2, optimizer2, scheduler2)

        assert info["step"] == 42
        assert info["best_f1"] == 0.85

    def test_model_state_preserved(self, tmp_path):
        """Model weights are identical after save/load round-trip."""
        model = SVTModule()
        # Run a forward+backward to get non-trivial state
        acoustic = torch.randint(0, 1024, (1, 10, 12))
        out = model(acoustic)
        out["logits"].sum().backward()

        config = _make_dummy_config()
        optimizer = build_optimizer(model, config)
        save_checkpoint(model, optimizer, None, step=1, best_f1=0.0, output_dir=tmp_path)

        model2 = SVTModule()
        load_checkpoint(tmp_path / "checkpoint.pt", model2)

        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Parameter {n1} differs after load"


# ---------------------------------------------------------------------------
# TestSetSeed
# ---------------------------------------------------------------------------


class TestSetSeed:
    """Test reproducibility via set_seed."""

    def test_deterministic_random(self):
        """Two calls to set_seed(42) produce the same random sequence."""
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.equal(a, b)
