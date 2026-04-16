"""SingMOS evaluation.

Target: 4.32 (Seen), paper Table II.
Wraps external SingMOS model (South-Twilight/SingMOS on HuggingFace).
Falls back gracefully if SingMOS is not installed.

Ticket: M4-07
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SINGMOS_MODEL_ID = "South-Twilight/SingMOS"


def load_singmos(device: str = "cuda"):
    """Load SingMOS model and processor from HuggingFace.

    Args:
        device: Device to load model on ("cuda" or "cpu").

    Returns:
        Tuple of (processor, model).

    Raises:
        ImportError: If transformers is not installed.
    """
    from transformers import AutoModelForAudioClassification, AutoProcessor

    processor = AutoProcessor.from_pretrained(SINGMOS_MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(SINGMOS_MODEL_ID).to(device)
    model.eval()
    return processor, model


def compute_singmos(
    wav_path: str,
    processor=None,
    model=None,
    device: str = "cuda",
) -> float | None:
    """Compute SingMOS score for a single WAV file.

    Falls back to None if SingMOS model is not available.

    Args:
        wav_path: Path to WAV file.
        processor: Pre-loaded SingMOS processor (optional, loads if None).
        model: Pre-loaded SingMOS model (optional, loads if None).
        device: Device for inference.

    Returns:
        SingMOS score (float) or None if model unavailable.
    """
    try:
        import torch

        if processor is None or model is None:
            processor, model = load_singmos(device=device)

        wav, sr = sf.read(wav_path, dtype="float32")

        inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        return float(logits.squeeze().cpu())
    except ImportError:
        logger.warning("SingMOS model not available. Install transformers to use.")
        return None
    except Exception as e:
        logger.warning("SingMOS computation failed for %s: %s", wav_path, e)
        return None


def evaluate_singmos(
    wav_dir: str | Path,
    device: str = "cuda",
) -> dict:
    """Evaluate SingMOS over a directory of WAV files.

    Args:
        wav_dir: Directory containing WAV files.
        device: Device for inference.

    Returns:
        dict with keys: "mean_singmos", "std_singmos", "n_samples", "per_file".
    """
    wav_dir = Path(wav_dir)
    wav_files = sorted(wav_dir.glob("*.wav"))

    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {wav_dir}")

    # Pre-load model once
    try:
        processor, model = load_singmos(device=device)
    except Exception as e:
        logger.error("Failed to load SingMOS model: %s", e)
        return {
            "mean_singmos": None,
            "std_singmos": None,
            "n_samples": 0,
            "per_file": [],
        }

    per_file = []
    for wav_path in wav_files:
        score = compute_singmos(
            str(wav_path), processor=processor, model=model, device=device
        )
        per_file.append({"filename": wav_path.stem, "singmos": score})
        if score is not None:
            logger.info("  %s: SingMOS = %.4f", wav_path.stem, score)

    valid_scores = [r["singmos"] for r in per_file if r["singmos"] is not None]

    if not valid_scores:
        return {
            "mean_singmos": None,
            "std_singmos": None,
            "n_samples": 0,
            "per_file": per_file,
        }

    return {
        "mean_singmos": float(np.mean(valid_scores)),
        "std_singmos": float(np.std(valid_scores)),
        "n_samples": len(valid_scores),
        "per_file": per_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute SingMOS scores")
    parser.add_argument("--syn_dir", type=str, required=True, help="Synthesized WAV directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--output", type=str, default="singmos_results.json", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results = evaluate_singmos(args.syn_dir, device=args.device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if results["mean_singmos"] is not None:
        logger.info(
            "SingMOS: %.4f +/- %.4f (n=%d)",
            results["mean_singmos"],
            results["std_singmos"],
            results["n_samples"],
        )
    else:
        logger.warning("SingMOS evaluation failed or model not available.")
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
