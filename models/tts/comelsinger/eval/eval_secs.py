"""SECS (Speaker Embedding Cosine Similarity) evaluation.

Target: 0.912 (Seen), paper Table II.
Uses WavLM (microsoft/wavlm-base-plus) for speaker embedding extraction.
Cosine similarity computed on mean-pooled last hidden state.

Ticket: M4-08
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

WAVLM_MODEL_ID = "microsoft/wavlm-base-plus"


def load_wavlm(model_name: str = WAVLM_MODEL_ID, device: str = "cuda"):
    """Load WavLM model and feature extractor from HuggingFace.

    Args:
        model_name: HuggingFace model ID.
        device: Device to load model on.

    Returns:
        Tuple of (feature_extractor, model).
    """
    from transformers import AutoFeatureExtractor, WavLMModel

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name).to(device)
    model.eval()
    return feature_extractor, model


def extract_speaker_embedding(
    wav: np.ndarray,
    sr: int,
    feature_extractor,
    model,
    device: str = "cuda",
):
    """Extract speaker embedding via mean pooling of WavLM last hidden state.

    Args:
        wav: Waveform array.
        sr: Sample rate.
        feature_extractor: WavLM feature extractor.
        model: WavLM model.
        device: Device for inference.

    Returns:
        Speaker embedding tensor of shape (1, D).
    """
    import torch

    inputs = feature_extractor(wav, sampling_rate=sr, return_tensors="pt").to(device)
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state  # (1, T, D)
    return hidden.mean(dim=1)  # (1, D)


def compute_secs(
    ref_wav: np.ndarray,
    syn_wav: np.ndarray,
    sr: int = 24000,
    feature_extractor=None,
    model=None,
    model_name: str = WAVLM_MODEL_ID,
    device: str = "cuda",
) -> float | None:
    """Compute speaker embedding cosine similarity.

    Uses WavLM for speaker embedding extraction.
    Falls back to None if transformers not available.

    Args:
        ref_wav: Reference waveform array.
        syn_wav: Synthesized waveform array.
        sr: Sample rate.
        feature_extractor: Pre-loaded feature extractor (optional).
        model: Pre-loaded WavLM model (optional).
        model_name: HuggingFace model ID (used if model not pre-loaded).
        device: Device for inference.

    Returns:
        Cosine similarity score (float) or None if model unavailable.
    """
    try:
        import torch
        import torch.nn.functional as F

        if feature_extractor is None or model is None:
            feature_extractor, model = load_wavlm(model_name, device)

        ref_emb = extract_speaker_embedding(ref_wav, sr, feature_extractor, model, device)
        syn_emb = extract_speaker_embedding(syn_wav, sr, feature_extractor, model, device)

        return float(F.cosine_similarity(ref_emb, syn_emb).cpu().item())
    except ImportError:
        logger.warning("transformers not available for SECS computation.")
        return None
    except Exception as e:
        logger.warning("SECS computation failed: %s", e)
        return None


def evaluate_secs(
    ref_dir: str | Path,
    syn_dir: str | Path,
    sr: int = 24000,
    model_name: str = WAVLM_MODEL_ID,
    device: str = "cuda",
) -> dict:
    """Evaluate SECS over directory of WAV files.

    Args:
        ref_dir: Directory containing reference WAV files.
        syn_dir: Directory containing synthesized WAV files.
        sr: Sample rate.
        model_name: HuggingFace model ID for WavLM.
        device: Device for inference.

    Returns:
        dict with keys: "mean_secs", "std_secs", "n_samples", "per_file".
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

    # Pre-load model once
    try:
        feature_extractor, model = load_wavlm(model_name, device)
    except Exception as e:
        logger.error("Failed to load WavLM model: %s", e)
        return {
            "mean_secs": None,
            "std_secs": None,
            "n_samples": 0,
            "per_file": [],
        }

    per_file = []
    for stem in common:
        ref_wav, _ = sf.read(str(ref_files[stem]), dtype="float32")
        syn_wav, _ = sf.read(str(syn_files[stem]), dtype="float32")
        score = compute_secs(
            ref_wav, syn_wav, sr=sr,
            feature_extractor=feature_extractor, model=model, device=device,
        )
        per_file.append({"filename": stem, "secs": score})
        if score is not None:
            logger.info("  %s: SECS = %.4f", stem, score)

    valid_scores = [r["secs"] for r in per_file if r["secs"] is not None]

    if not valid_scores:
        return {
            "mean_secs": None,
            "std_secs": None,
            "n_samples": 0,
            "per_file": per_file,
        }

    return {
        "mean_secs": float(np.mean(valid_scores)),
        "std_secs": float(np.std(valid_scores)),
        "n_samples": len(valid_scores),
        "per_file": per_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute SECS (Speaker Embedding Cosine Similarity)")
    parser.add_argument("--ref_dir", type=str, required=True, help="Reference WAV directory")
    parser.add_argument("--syn_dir", type=str, required=True, help="Synthesized WAV directory")
    parser.add_argument("--sr", type=int, default=24000, help="Sample rate")
    parser.add_argument("--model_name", type=str, default=WAVLM_MODEL_ID, help="WavLM model ID")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--output", type=str, default="secs_results.json", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results = evaluate_secs(
        args.ref_dir, args.syn_dir, sr=args.sr,
        model_name=args.model_name, device=args.device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    if results["mean_secs"] is not None:
        logger.info(
            "SECS: %.4f +/- %.4f (n=%d)",
            results["mean_secs"],
            results["std_secs"],
            results["n_samples"],
        )
    else:
        logger.warning("SECS evaluation failed or model not available.")
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
