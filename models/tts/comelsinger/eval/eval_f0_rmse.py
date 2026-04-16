"""F0-RMSE evaluation (semitone scale).

Target: 0.042 (Seen), paper Table III.
Uses pyworld for F0 extraction, RMSE on voiced frames only.
F0 converted to semitone scale: 12 * log2(f0 / 440) + 69.

Ticket: M4-06
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np

from models.tts.comelsinger.eval._utils import load_wav, match_wav_pairs

logger = logging.getLogger(__name__)


def hz_to_semitone(f0: np.ndarray) -> np.ndarray:
    """Convert Hz to semitone (A4=440Hz reference).

    Unvoiced frames (f0 <= 0) are returned as 0.

    Args:
        f0: F0 array in Hz.

    Returns:
        Semitone array.
    """
    voiced = f0 > 0
    semitone = np.zeros_like(f0)
    semitone[voiced] = 12.0 * np.log2(f0[voiced] / 440.0) + 69.0
    return semitone


def compute_f0_rmse(
    ref_wav: np.ndarray,
    syn_wav: np.ndarray,
    sr: int = 24000,
) -> float:
    """Compute F0-RMSE excluding unvoiced frames.

    Uses pyworld harvest for F0 extraction. RMSE is computed on
    semitone-converted F0 values where both reference and synthesis
    are voiced.

    Args:
        ref_wav: Reference waveform array.
        syn_wav: Synthesized waveform array.
        sr: Sample rate.

    Returns:
        F0-RMSE in semitones. Returns NaN if no mutually voiced frames exist.
    """
    try:
        import pyworld as pw
    except ImportError:
        raise ImportError(
            "pyworld is required for F0-RMSE computation. "
            "Install with: uv add pyworld"
        )

    ref_f0, _ = pw.harvest(ref_wav.astype(np.float64), sr)
    syn_f0, _ = pw.harvest(syn_wav.astype(np.float64), sr)

    # Align lengths (take minimum)
    min_len = min(len(ref_f0), len(syn_f0))
    ref_f0 = ref_f0[:min_len]
    syn_f0 = syn_f0[:min_len]

    # Only evaluate on mutually voiced frames
    voiced = (ref_f0 > 0) & (syn_f0 > 0)
    if voiced.sum() == 0:
        return float("nan")

    ref_st = hz_to_semitone(ref_f0)[voiced]
    syn_st = hz_to_semitone(syn_f0)[voiced]

    rmse = float(np.sqrt(np.mean((ref_st - syn_st) ** 2)))
    return rmse


def evaluate_f0_rmse(
    ref_dir: str | Path,
    syn_dir: str | Path,
    sr: int = 24000,
) -> dict:
    """Evaluate F0-RMSE over directory of WAV files.

    Args:
        ref_dir: Directory containing reference WAV files.
        syn_dir: Directory containing synthesized WAV files.
        sr: Sample rate.

    Returns:
        dict with keys: "mean_f0_rmse", "std_f0_rmse", "n_samples", "per_file".
    """
    pairs = match_wav_pairs(ref_dir, syn_dir)

    per_file = []
    for ref_path, syn_path in pairs:
        ref_wav = load_wav(ref_path, sr=sr)
        syn_wav = load_wav(syn_path, sr=sr)
        rmse = compute_f0_rmse(ref_wav, syn_wav, sr=sr)
        per_file.append({"filename": ref_path.stem, "f0_rmse": rmse})
        logger.info("  %s: F0-RMSE = %.6f", ref_path.stem, rmse)

    # Exclude NaN entries from aggregation
    valid_scores = [r["f0_rmse"] for r in per_file if not math.isnan(r["f0_rmse"])]

    if not valid_scores:
        return {
            "mean_f0_rmse": float("nan"),
            "std_f0_rmse": float("nan"),
            "n_samples": 0,
            "per_file": per_file,
        }

    return {
        "mean_f0_rmse": float(np.mean(valid_scores)),
        "std_f0_rmse": float(np.std(valid_scores)),
        "n_samples": len(valid_scores),
        "per_file": per_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute F0-RMSE (semitone scale)")
    parser.add_argument("--ref_dir", type=str, required=True, help="Reference WAV directory")
    parser.add_argument("--syn_dir", type=str, required=True, help="Synthesized WAV directory")
    parser.add_argument("--sr", type=int, default=24000, help="Sample rate")
    parser.add_argument("--output", type=str, default="f0_rmse_results.json", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results = evaluate_f0_rmse(args.ref_dir, args.syn_dir, sr=args.sr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(
        "F0-RMSE: %.6f +/- %.6f (n=%d)",
        results["mean_f0_rmse"],
        results["std_f0_rmse"],
        results["n_samples"],
    )
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
