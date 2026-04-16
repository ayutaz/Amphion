"""Shared utilities for evaluation scripts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def match_wav_pairs(
    ref_dir: str | Path, syn_dir: str | Path
) -> list[tuple[Path, Path]]:
    """Match WAV files by stem name between ref and syn directories.

    Args:
        ref_dir: Directory containing reference WAV files.
        syn_dir: Directory containing synthesized WAV files.

    Returns:
        Sorted list of (ref_path, syn_path) tuples for matching stems.

    Raises:
        FileNotFoundError: If no matching WAV files are found.
    """
    ref_dir, syn_dir = Path(ref_dir), Path(syn_dir)
    ref_files = {p.stem: p for p in sorted(ref_dir.glob("*.wav"))}
    syn_files = {p.stem: p for p in sorted(syn_dir.glob("*.wav"))}
    common = sorted(set(ref_files) & set(syn_files))
    if not common:
        raise FileNotFoundError(
            f"No matching WAV files between {ref_dir} and {syn_dir}"
        )
    return [(ref_files[k], syn_files[k]) for k in common]


def load_wav(path: str | Path, sr: int = 24000) -> np.ndarray:
    """Load WAV file and resample if needed.

    Args:
        path: Path to WAV file.
        sr: Target sample rate.

    Returns:
        Waveform array at the target sample rate.
    """
    wav, file_sr = sf.read(str(path), dtype="float32")
    if file_sr != sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return wav
