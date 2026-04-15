"""Shared utilities for CoMelSinger training scripts.

Extracted from train_svt.py and train_s2a.py to avoid code duplication.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to .yaml config file.

    Returns:
        Nested dict with all configuration values.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If required top-level keys are missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    required_keys = {"model", "training", "optimizer", "loss", "data", "checkpoint"}
    missing = required_keys - set(config.keys())
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    return config


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def check_loss_finite(loss: torch.Tensor, step: int) -> bool:
    """Check whether a loss value is finite (not NaN or Inf).

    Args:
        loss: scalar loss tensor.
        step: current training step (for log messages).

    Returns:
        True if loss is finite, False otherwise.
    """
    if not torch.isfinite(loss):
        logger.warning(
            "Non-finite loss detected at step %d: %s", step, loss.item()
        )
        return False
    return True


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Safely unwrap DDP/PeftModel wrappers.

    Traverses `.module` attributes (DDP wrappers) and then tries
    `.get_base_model()` (PEFT wrappers) to reach the underlying model.

    Args:
        model: possibly wrapped model.

    Returns:
        The innermost unwrapped model.
    """
    m = model
    while hasattr(m, "module"):
        m = m.module
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    return m
