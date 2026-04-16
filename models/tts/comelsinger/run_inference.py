"""Batch inference CLI for CoMelSinger.

Reads a testset JSON and runs the CoMelSinger inference pipeline on each
sample, saving generated WAV files and metadata.

Usage:
    python -m models.tts.comelsinger.run_inference \
        --config configs/comelsinger/inference.yaml \
        --testset data/testset.json \
        --output_dir output/generated \
        --device cuda

Testset JSON format:
    [
        {
            "id": "sample_001",
            "wav_path": "path/to/reference.wav",
            "text": "ni hao shi jie",
            "pitch_sequence": [60, 62, 64, 67],
            "note_durations": [0.5, 0.5, 0.5, 0.5],
            "speaker_id": 0
        },
        ...
    ]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CoMelSinger batch inference CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to inference config YAML/JSON (for from_pretrained).",
    )
    parser.add_argument(
        "--testset",
        type=Path,
        required=True,
        help="Path to testset JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save generated WAV files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device (cpu, cuda, cuda:0, etc.).",
    )
    parser.add_argument(
        "--lora_path",
        type=Path,
        default=None,
        help="Optional path to LoRA adapter weights directory.",
    )
    parser.add_argument(
        "--ablation_tag",
        type=str,
        default=None,
        help="Ablation experiment tag (appended to output subdir).",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--rescale_cfg",
        type=float,
        default=0.75,
        help="CFG rescaling factor.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="zh",
        help="Language code for G2P (zh, en).",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000,
        help="Output WAV sample rate.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Testset loading
# ---------------------------------------------------------------------------


def load_testset(testset_path: Path) -> list[dict[str, Any]]:
    """Load testset from JSON file.

    Supports both JSON array format and JSONL format.

    Args:
        testset_path: Path to testset file.

    Returns:
        List of sample dictionaries.

    Raises:
        FileNotFoundError: If testset file does not exist.
        ValueError: If testset format is invalid.
    """
    if not testset_path.exists():
        raise FileNotFoundError(f"Testset not found: {testset_path}")

    with open(testset_path, encoding="utf-8") as f:
        content = f.read().strip()

    # Try JSON array first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        raise ValueError("Expected JSON array at top level")
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL
    entries = []
    for i, line in enumerate(content.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {i} of {testset_path}: {exc}"
            ) from exc

    if not entries:
        raise ValueError(f"Testset is empty: {testset_path}")

    return entries


def validate_sample(sample: dict[str, Any], index: int) -> None:
    """Validate a single testset sample has required fields.

    Args:
        sample: Sample dictionary.
        index: Sample index (for error messages).

    Raises:
        ValueError: If required fields are missing.
    """
    required = ["wav_path", "text", "pitch_sequence", "note_durations"]
    missing = [k for k in required if k not in sample]
    if missing:
        raise ValueError(
            f"Sample {index}: missing required fields: {missing}"
        )
    if len(sample["pitch_sequence"]) != len(sample["note_durations"]):
        raise ValueError(
            f"Sample {index}: pitch_sequence length ({len(sample['pitch_sequence'])}) "
            f"!= note_durations length ({len(sample['note_durations'])})"
        )


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------


def run_inference(
    pipeline,
    samples: list[dict[str, Any]],
    output_dir: Path,
    language: str = "zh",
    cfg_scale: float = 2.5,
    rescale_cfg: float = 0.75,
    sample_rate: int = 24000,
) -> dict[str, Any]:
    """Run inference on all samples and save results.

    Args:
        pipeline: CoMelSingerInferencePipeline instance.
        samples: List of testset sample dicts.
        output_dir: Directory to write WAV files.
        language: Language code for G2P.
        cfg_scale: CFG scale.
        rescale_cfg: CFG rescaling factor.
        sample_rate: Output sample rate.

    Returns:
        Metadata dict with results and statistics.
    """
    import soundfile as sf

    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to use tqdm for progress display
    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(samples), total=len(samples), desc="Generating")
    except ImportError:
        iterator = enumerate(samples)
        logger.info("tqdm not available; running without progress bar.")

    results: list[dict[str, Any]] = []
    n_ok = 0
    n_skip = 0
    total_time = 0.0

    for i, sample in iterator:
        sample_id = sample.get("id", f"sample_{i:04d}")
        try:
            validate_sample(sample, i)

            t_start = time.time()

            audio = pipeline.synthesize(
                prompt_wav_path=sample["wav_path"],
                lyrics=sample["text"],
                pitch_sequence=sample["pitch_sequence"],
                note_durations=sample["note_durations"],
                language=language,
                cfg_scale=cfg_scale,
                rescale_cfg=rescale_cfg,
            )

            elapsed = time.time() - t_start
            total_time += elapsed

            # Save WAV
            wav_filename = f"{sample_id}.wav"
            wav_path = output_dir / wav_filename
            sf.write(str(wav_path), audio, sample_rate)

            result = {
                "id": sample_id,
                "output_path": str(wav_path),
                "duration_sec": len(audio) / sample_rate,
                "inference_time_sec": round(elapsed, 3),
                "status": "ok",
            }
            if "speaker_id" in sample:
                result["speaker_id"] = sample["speaker_id"]
            results.append(result)
            n_ok += 1

            logger.info(
                "Generated %s (%.2fs audio in %.2fs)",
                sample_id,
                len(audio) / sample_rate,
                elapsed,
            )

        except NotImplementedError as exc:
            # Model not loaded — this is expected in stub mode
            logger.error("Skipping %s: %s", sample_id, exc)
            results.append({
                "id": sample_id,
                "status": "skip",
                "error": str(exc),
            })
            n_skip += 1

        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: %s", sample_id, exc)
            results.append({
                "id": sample_id,
                "status": "error",
                "error": str(exc),
            })
            n_skip += 1

    metadata = {
        "n_total": len(samples),
        "n_ok": n_ok,
        "n_skip": n_skip,
        "total_inference_time_sec": round(total_time, 3),
        "samples": results,
    }
    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for batch inference CLI."""
    args = parse_args()

    # Resolve output directory with optional ablation tag
    output_dir = args.output_dir
    if args.ablation_tag:
        output_dir = output_dir / args.ablation_tag

    # Load testset
    logger.info("Loading testset from %s", args.testset)
    samples = load_testset(args.testset)
    logger.info("Loaded %d samples", len(samples))

    # Build pipeline
    from models.tts.comelsinger.comelsinger_inference import (
        CoMelSingerInferencePipeline,
    )

    if args.config is not None:
        try:
            pipeline = CoMelSingerInferencePipeline.from_pretrained(
                config_path=args.config,
                lora_path=args.lora_path,
                device=args.device,
            )
        except NotImplementedError:
            logger.warning(
                "from_pretrained not yet implemented. "
                "Creating pipeline with default PitchTokenizer only."
            )
            pipeline = CoMelSingerInferencePipeline(device=args.device)
    else:
        logger.info(
            "No config provided. Creating pipeline with default PitchTokenizer only."
        )
        pipeline = CoMelSingerInferencePipeline(device=args.device)

    # Optional LoRA loading
    if args.lora_path is not None:
        try:
            pipeline.load_lora_weights(args.lora_path)
        except (NotImplementedError, ImportError) as exc:
            logger.warning("Could not load LoRA weights: %s", exc)

    # Run inference
    logger.info("Starting inference, output_dir=%s", output_dir)
    metadata = run_inference(
        pipeline=pipeline,
        samples=samples,
        output_dir=output_dir,
        language=args.language,
        cfg_scale=args.cfg_scale,
        rescale_cfg=args.rescale_cfg,
        sample_rate=args.sample_rate,
    )

    # Save metadata
    meta_path = output_dir / "metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(
        "Done. generated=%d  skipped=%d  metadata=%s",
        metadata["n_ok"],
        metadata["n_skip"],
        meta_path,
    )


if __name__ == "__main__":
    main()
