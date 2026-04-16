"""Shared test fixtures for CoMelSinger tests."""
import numpy as np
import pytest
import torch


@pytest.fixture
def sine_wave_24k():
    """Generate 2-second 440Hz sine wave at 24kHz."""
    sr = 24000
    t = np.linspace(0, 2.0, sr * 2, dtype=np.float32)
    return np.sin(2 * np.pi * 440.0 * t)


@pytest.fixture
def dummy_acoustic_tokens():
    """Generate dummy acoustic tokens (B=2, T=50, 12 codebooks)."""
    return torch.randint(0, 1024, (2, 50, 12))


@pytest.fixture
def dummy_batch():
    """Generate a dummy collated batch."""
    B, T = 4, 50
    return {
        "acoustic_tokens": torch.randint(0, 1024, (B, 12, T)),
        "semantic_tokens": torch.randint(0, 1024, (B, T)),
        "pitch_tokens": torch.randint(0, 129, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "speaker_id": torch.tensor([0, 0, 1, 1]),
        "note_durations": [torch.tensor([10, 10, 10, 10, 10])] * B,
        "note_pitches": [torch.tensor([60, 62, 64, 67, 69])] * B,
    }
