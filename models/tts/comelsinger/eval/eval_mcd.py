"""MCD (Mel Cepstral Distortion) evaluation.

Target: 4.17 dB (Seen), paper Table II.
Uses MFCC 1-12 (0th excluded) with DTW alignment.
Standard MCD formula: (10/ln10) * sqrt(2) * mean(sqrt(sum(diff^2)))

Ticket: M4-05
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def compute_mcd(
    ref_wav: np.ndarray,
    syn_wav: np.ndarray,
    sr: int = 24000,
    n_mfcc: int = 13,
) -> float:
    """Compute MCD in dB using DTW alignment.

    MFCC 1-12 (0th excluded), DTW for alignment, standard MCD formula.

    Args:
        ref_wav: Reference waveform array.
        syn_wav: Synthesized waveform array.
        sr: Sample rate.
        n_mfcc: Number of MFCCs to extract (0th will be excluded).

    Returns:
        MCD value in dB.
    """
    # Extract MFCCs, exclude 0th coefficient (energy)
    ref_mfcc = librosa.feature.mfcc(y=ref_wav.astype(np.float32), sr=sr, n_mfcc=n_mfcc)[1:]  # (12, T_ref)
    syn_mfcc = librosa.feature.mfcc(y=syn_wav.astype(np.float32), sr=sr, n_mfcc=n_mfcc)[1:]  # (12, T_syn)

    # DTW alignment
    try:
        from dtw import dtw as dtw_func

        alignment = dtw_func(ref_mfcc.T, syn_mfcc.T)
        ref_aligned = ref_mfcc[:, alignment.index1]
        syn_aligned = syn_mfcc[:, alignment.index2]
    except ImportError:
        # Fallback: simple Euclidean DTW using librosa (scipy-based)
        logger.warning(
            "dtw-python not installed. Using librosa DTW as fallback. "
            "Install with: uv add dtw-python"
        )
        D, wp = librosa.sequence.dtw(ref_mfcc, syn_mfcc, metric="euclidean")
        # wp is (N, 2) with (ref_idx, syn_idx), reversed order
        wp = wp[::-1]  # reverse to ascending order
        ref_aligned = ref_mfcc[:, wp[:, 0]]
        syn_aligned = syn_mfcc[:, wp[:, 1]]

    # MCD formula: (10/ln10) * sqrt(2) * mean(sqrt(sum(diff^2, axis=0)))
    diff = ref_aligned - syn_aligned
    frame_dist = np.sqrt(np.sum(diff**2, axis=0))
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2.0) * np.mean(frame_dist)
    return float(mcd)


def evaluate_mcd(
    ref_dir: str | Path,
    syn_dir: str | Path,
    sr: int = 24000,
) -> dict:
    """Evaluate MCD over directory of WAV files.

    Matches files by filename (stem). Skips files that exist in only one directory.

    Args:
        ref_dir: Directory containing reference WAV files.
        syn_dir: Directory containing synthesized WAV files.
        sr: Sample rate.

    Returns:
        dict with keys: "mean_mcd", "std_mcd", "n_samples", "per_file".
    """
    ref_dir = Path(ref_dir)
    syn_dir = Path(syn_dir)

    ref_files = {p.stem: p for p in sorted(ref_dir.glob("*.wav"))}
    syn_files = {p.stem: p for p in sorted(syn_dir.glob("*.wav"))}
    common = sorted(set(ref_files.keys()) & set(syn_files.keys()))

    if not common:
        raise FileNotFoundError(
            f"No matching WAV files found between {ref_dir} and {syn_dir}"
        )

    per_file = []
    for stem in common:
        ref_wav, _ = sf.read(str(ref_files[stem]), dtype="float32")
        syn_wav, _ = sf.read(str(syn_files[stem]), dtype="float32")
        mcd = compute_mcd(ref_wav, syn_wav, sr=sr)
        per_file.append({"filename": stem, "mcd": mcd})
        logger.info("  %s: MCD = %.4f dB", stem, mcd)

    scores = [r["mcd"] for r in per_file]
    return {
        "mean_mcd": float(np.mean(scores)),
        "std_mcd": float(np.std(scores)),
        "n_samples": len(scores),
        "per_file": per_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute MCD (Mel Cepstral Distortion)")
    parser.add_argument("--ref_dir", type=str, required=True, help="Reference WAV directory")
    parser.add_argument("--syn_dir", type=str, required=True, help="Synthesized WAV directory")
    parser.add_argument("--sr", type=int, default=24000, help="Sample rate")
    parser.add_argument("--output", type=str, default="mcd_results.json", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results = evaluate_mcd(args.ref_dir, args.syn_dir, sr=args.sr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("MCD: %.4f +/- %.4f dB (n=%d)", results["mean_mcd"], results["std_mcd"], results["n_samples"])
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
