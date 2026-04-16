"""SVT F1 evaluation.

Target: F1 = 0.711, paper Table III.
Evaluates pitch prediction accuracy via frozen SVT module.
Pipeline: acoustic tokens -> frozen SVT -> predicted pitch tokens -> F1 vs GT.

Ticket: M4-09
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_svt_f1(
    acoustic_tokens: torch.Tensor,
    gt_pitch_tokens: torch.Tensor,
    svt_model,
    device: str = "cuda",
) -> dict:
    """Compute frame-level pitch prediction F1 via frozen SVT.

    Args:
        acoustic_tokens: (B, T, num_codebooks) or (T, num_codebooks) acoustic tokens.
        gt_pitch_tokens: (B, T) or (T,) ground truth pitch tokens.
        svt_model: SVTModule instance (will be set to eval mode).
        device: Device for inference.

    Returns:
        dict with keys: "precision", "recall", "f1".
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    svt_model.eval()
    svt_model.to(device)

    # Add batch dimension if needed
    if acoustic_tokens.dim() == 2:
        acoustic_tokens = acoustic_tokens.unsqueeze(0)  # (1, T, C)
    if gt_pitch_tokens.dim() == 1:
        gt_pitch_tokens = gt_pitch_tokens.unsqueeze(0)  # (1, T)

    with torch.no_grad():
        output = svt_model(acoustic_tokens.to(device))
        pred_logits = output["logits"]  # (B, T, pitch_vocab_size)
        pred_tokens = pred_logits.argmax(dim=-1).cpu()  # (B, T)

    # Flatten across batch
    pred_flat = pred_tokens.view(-1).numpy()
    gt_flat = gt_pitch_tokens.view(-1).numpy()

    # Exclude padding/unvoiced (token 0) from evaluation
    mask = gt_flat != 0
    if mask.sum() == 0:
        logger.warning("No voiced frames found in ground truth; returning 0.0 for all metrics.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    gt_masked = gt_flat[mask]
    pred_masked = pred_flat[mask]

    precision = float(precision_score(gt_masked, pred_masked, average="macro", zero_division=0))
    recall = float(recall_score(gt_masked, pred_masked, average="macro", zero_division=0))
    f1 = float(f1_score(gt_masked, pred_masked, average="macro", zero_division=0))

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_svt_f1(
    syn_tokens_dir: str | Path,
    gt_tokens_dir: str | Path,
    svt_ckpt: str | Path,
    device: str = "cuda",
) -> dict:
    """Evaluate SVT F1 over directory of token files.

    Token files are expected to be .pt files containing dicts with
    "acoustic_tokens" and "pitch_tokens" keys.

    Args:
        syn_tokens_dir: Directory with synthesized acoustic token .pt files.
        gt_tokens_dir: Directory with ground truth pitch token .pt files.
        svt_ckpt: Path to SVT model checkpoint.
        device: Device for inference.

    Returns:
        dict with keys: "mean_f1", "std_f1", "n_samples", "per_file".
    """
    from models.tts.comelsinger.svt_module import SVTModule

    syn_dir = Path(syn_tokens_dir)
    gt_dir = Path(gt_tokens_dir)

    # Load SVT model
    svt_model = SVTModule()
    state_dict = torch.load(str(svt_ckpt), map_location="cpu")
    # Handle checkpoint wrapped in {"model_state_dict": ...}
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    svt_model.load_state_dict(state_dict)
    svt_model.freeze()
    svt_model.to(device)

    syn_files = {p.stem: p for p in sorted(syn_dir.glob("*.pt"))}
    gt_files = {p.stem: p for p in sorted(gt_dir.glob("*.pt"))}
    common = sorted(set(syn_files.keys()) & set(gt_files.keys()))

    if not common:
        raise FileNotFoundError(
            f"No matching .pt files found between {syn_dir} and {gt_dir}"
        )

    per_file = []
    for stem in common:
        syn_data = torch.load(str(syn_files[stem]), map_location="cpu")
        gt_data = torch.load(str(gt_files[stem]), map_location="cpu")

        acoustic_tokens = syn_data["acoustic_tokens"]  # (T, num_codebooks)
        gt_pitch_tokens = gt_data["pitch_tokens"]  # (T,)

        result = compute_svt_f1(acoustic_tokens, gt_pitch_tokens, svt_model, device)
        per_file.append({"filename": stem, **result})
        logger.info("  %s: F1 = %.4f", stem, result["f1"])

    f1_scores = [r["f1"] for r in per_file]
    return {
        "mean_f1": float(np.mean(f1_scores)),
        "std_f1": float(np.std(f1_scores)),
        "n_samples": len(f1_scores),
        "per_file": per_file,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute SVT F1 score")
    parser.add_argument("--syn_dir", type=str, required=True, help="Synthesized acoustic tokens directory (.pt)")
    parser.add_argument("--gt_tokens_dir", type=str, required=True, help="GT pitch tokens directory (.pt)")
    parser.add_argument("--svt_ckpt", type=str, required=True, help="SVT model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--output", type=str, default="svt_f1_results.json", help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results = evaluate_svt_f1(
        args.syn_dir, args.gt_tokens_dir, args.svt_ckpt, device=args.device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(
        "SVT F1: %.4f +/- %.4f (n=%d)",
        results["mean_f1"],
        results["std_f1"],
        results["n_samples"],
    )
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
