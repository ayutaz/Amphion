"""CoMelSinger data preprocessing CLI.

Reads a dataset manifest (JSONL), extracts acoustic/semantic/pitch/phone features
for each sample, and writes .pt files to output_dir.

Usage:
    python -m models.tts.comelsinger.run_preprocess \
        --data_dir /path/to/dataset \
        --output_dir /path/to/output \
        --manifest manifest.jsonl \
        --device cuda

Manifest format (one JSON object per line):
    {
        "id": "song001_phrase01",
        "audio_path": "wavs/song001_phrase01.wav",
        "text": "你好世界",
        "note_midi": [60, 62, 64, 65],
        "note_duration": [4, 4, 4, 4],
        "speaker_id": 0
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

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
    parser = argparse.ArgumentParser(
        description="CoMelSinger data preprocessing pipeline"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Root directory containing the raw dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where .pt feature files will be written.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="manifest.jsonl",
        help="JSONL manifest filename relative to --data_dir (default: manifest.jsonl).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Computation device (default: cpu).",
    )
    parser.add_argument(
        "--sr_acoustic",
        type=int,
        default=24000,
        help="Sample rate for acoustic token extraction (default: 24000).",
    )
    parser.add_argument(
        "--sr_semantic",
        type=int,
        default=16000,
        help="Sample rate for semantic token extraction (default: 16000).",
    )
    parser.add_argument(
        "--num_codebooks",
        type=int,
        default=12,
        help="Number of RVQ codebooks (default: 12).",
    )
    parser.add_argument(
        "--phone_vocab",
        type=Path,
        default=None,
        help="Path to phone2id JSON file. If omitted, a simple character-level mapping is used.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(device: torch.device, cfg: dict | None = None):
    """Load all pretrained models required for preprocessing.

    Loads Amphion CodecEncoder/Decoder, w2v-bert-2.0, RepCodec semantic
    quantizer, normalization stats, and a PitchTokenizer.

    Args:
        device: Computation device.
        cfg: Optional dict with custom checkpoint paths. Supported keys:
            - codec_encoder_ckpt: path to acoustic codec encoder safetensors
            - codec_decoder_ckpt: path to acoustic codec decoder safetensors
            - semantic_codec_ckpt: path to semantic codec safetensors
            - w2v_bert_model: HuggingFace model ID or local path
            - w2v_stats: path to wav2vec2bert_stats.pt

    Returns:
        dict with keys: codec_encoder, codec_decoder,
                        w2v_bert_model, semantic_codec,
                        semantic_mean, semantic_std, pitch_tokenizer
    """
    import safetensors.torch
    from safetensors import safe_open

    from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer
    from models.tts.maskgct.maskgct_utils import build_acoustic_codec, build_semantic_codec
    from transformers import Wav2Vec2BertModel
    from utils.util import load_config

    if cfg is None:
        cfg = {}

    maskgct_cfg = load_config("./models/tts/maskgct/config/maskgct.json")

    # Acoustic codec (encoder + decoder)
    codec_enc, codec_dec = build_acoustic_codec(
        maskgct_cfg.model.acoustic_codec, device
    )
    safetensors.torch.load_model(
        codec_enc,
        cfg.get(
            "codec_encoder_ckpt",
            "models/tts/maskgct/ckpt/amphion_maskgct/acoustic_codec/model.safetensors",
        ),
    )
    # CodecDecoder: shared tensor workaround via safe_open
    dec_weights: dict = {}
    dec_path = cfg.get(
        "codec_decoder_ckpt",
        "models/tts/maskgct/ckpt/amphion_maskgct/acoustic_codec/model_1.safetensors",
    )
    with safe_open(dec_path, framework="pt") as f:
        for k in f.keys():
            dec_weights[k] = f.get_tensor(k)
    codec_dec.load_state_dict(dec_weights, strict=False)

    # Semantic model (w2v-bert-2.0)
    w2v_model_id = cfg.get("w2v_bert_model", "facebook/w2v-bert-2.0")
    semantic_model = Wav2Vec2BertModel.from_pretrained(w2v_model_id)
    semantic_model.to(device).eval()

    # Semantic codec (RepCodec)
    semantic_codec = build_semantic_codec(maskgct_cfg.model.semantic_codec, device)
    safetensors.torch.load_model(
        semantic_codec,
        cfg.get(
            "semantic_codec_ckpt",
            "models/tts/maskgct/ckpt/amphion_maskgct/semantic_codec/model.safetensors",
        ),
    )

    # Normalization stats
    stats = torch.load(
        cfg.get("w2v_stats", "models/tts/maskgct/ckpt/wav2vec2bert_stats.pt"),
        map_location="cpu",
        weights_only=True,
    )

    return {
        "codec_encoder": codec_enc.to(device).eval(),
        "codec_decoder": codec_dec.to(device).eval(),
        "w2v_bert_model": semantic_model,
        "semantic_codec": semantic_codec.to(device).eval(),
        "semantic_mean": stats["mean"].to(device),
        "semantic_std": stats["var"].sqrt().to(device),
        "pitch_tokenizer": PitchTokenizer(),
    }


# ---------------------------------------------------------------------------
# Pitch tokenizer
# ---------------------------------------------------------------------------

def load_pitch_tokenizer():
    """Return a PitchTokenizer instance."""
    from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer
    return PitchTokenizer()


# ---------------------------------------------------------------------------
# Phone vocabulary
# ---------------------------------------------------------------------------

def load_phone2id(phone_vocab_path: Path | None) -> dict:
    """Load or build a phone-to-ID mapping.

    If phone_vocab_path is provided, loads from JSON file.
    Otherwise falls back to a minimal default mapping.
    """
    if phone_vocab_path is not None and phone_vocab_path.exists():
        with open(phone_vocab_path, encoding="utf-8") as f:
            return json.load(f)
    logger.warning(
        "No phone vocabulary file provided; using empty mapping. "
        "All phones will map to <unk> (id=0)."
    )
    return {"<unk>": 0}


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def process_one(
    entry: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    models: dict,
    pitch_tokenizer,
    phone2id: dict,
    device: torch.device,
    sr_acoustic: int = 24000,
    sr_semantic: int = 16000,
) -> dict[str, Any]:
    """Extract all features for a single manifest entry and save to .pt.

    Args:
        entry: Manifest row with keys: id, audio_path, text,
               note_midi, note_duration, speaker_id
        data_dir: Dataset root (audio_path is relative to this)
        output_dir: Where to write the .pt file
        models: Dict returned by load_models()
        pitch_tokenizer: PitchTokenizer instance
        phone2id: Phone-to-ID mapping
        device: Computation device
        sr_acoustic: Sample rate for acoustic extraction
        sr_semantic: Sample rate for semantic extraction

    Returns:
        Metadata dict (id + output path) for the saved sample.
    """
    import torchaudio

    from models.tts.comelsinger.preprocess import (
        extract_acoustic_tokens,
        extract_pitch_tokens,
        extract_phone_ids,
        extract_semantic_tokens,
        save_preprocessed,
    )

    sample_id = entry["id"]
    audio_path = data_dir / entry["audio_path"]

    # --- Load waveform ---
    waveform, orig_sr = torchaudio.load(str(audio_path))
    waveform = waveform.squeeze(0)  # (T,)

    # Resample for acoustic extraction (24kHz)
    if orig_sr != sr_acoustic:
        waveform_acoustic = torchaudio.functional.resample(
            waveform, orig_sr, sr_acoustic
        )
    else:
        waveform_acoustic = waveform

    # Resample for semantic extraction (16kHz)
    if orig_sr != sr_semantic:
        waveform_semantic = torchaudio.functional.resample(
            waveform, orig_sr, sr_semantic
        )
    else:
        waveform_semantic = waveform

    # --- Acoustic tokens ---
    acoustic_tokens = extract_acoustic_tokens(
        waveform_acoustic,
        models["codec_encoder"],
        models["codec_decoder"],
        device=device,
    )
    T_a = acoustic_tokens.shape[0]

    # --- Semantic tokens ---
    semantic_tokens = extract_semantic_tokens(
        waveform_semantic,
        models["w2v_bert_model"],
        models["semantic_codec"],
        models["semantic_mean"],
        models["semantic_std"],
        target_len=T_a,
        device=device,
    )

    # --- Pitch tokens ---
    pitch_tokens = extract_pitch_tokens(
        waveform_acoustic,
        pitch_tokenizer,
        target_len=T_a,
        sr=sr_acoustic,
    )

    # --- Phone IDs ---
    phone_ids = extract_phone_ids(entry["text"], phone2id, language="zh")

    # --- Score tensors ---
    note_durations = torch.tensor(entry["note_duration"], dtype=torch.long)
    note_pitches = torch.tensor(entry["note_midi"], dtype=torch.long)

    # --- Attention mask (all valid, no padding at this stage) ---
    attention_mask = torch.ones(T_a, dtype=torch.long)

    # --- Save ---
    out_path = output_dir / f"{sample_id}.pt"
    save_preprocessed(
        output_path=out_path,
        acoustic_tokens=acoustic_tokens,
        semantic_tokens=semantic_tokens,
        pitch_tokens=pitch_tokens,
        phone_ids=phone_ids,
        note_durations=note_durations,
        note_pitches=note_pitches,
        attention_mask=attention_mask,
        speaker_id=int(entry.get("speaker_id", 0)),
    )

    return {"id": sample_id, "path": str(out_path), "T_a": T_a}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    manifest_path = args.data_dir / args.manifest
    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    # Load manifest
    entries: list[dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    logger.info("Loaded %d entries from %s", len(entries), manifest_path)

    logger.info("Loading models onto %s ...", device)
    models = load_models(device)

    pitch_tokenizer = models.get("pitch_tokenizer") or load_pitch_tokenizer()
    phone2id = load_phone2id(args.phone_vocab)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each sample
    try:
        from tqdm import tqdm
        iterator = tqdm(entries, desc="Preprocessing")
    except ImportError:
        iterator = entries
        logger.info("tqdm not available; running without progress bar.")

    metadata: list[dict] = []
    n_ok = 0
    n_skip = 0

    for entry in iterator:
        sample_id = entry.get("id", "<unknown>")
        try:
            meta = process_one(
                entry=entry,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                models=models,
                pitch_tokenizer=pitch_tokenizer,
                phone2id=phone2id,
                device=device,
                sr_acoustic=args.sr_acoustic,
                sr_semantic=args.sr_semantic,
            )
            metadata.append(meta)
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s: %s", sample_id, exc)
            n_skip += 1

    # Write metadata index
    meta_path = args.output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"samples": metadata, "n_ok": n_ok, "n_skip": n_skip}, f, indent=2)

    logger.info(
        "Done. processed=%d  skipped=%d  metadata=%s",
        n_ok, n_skip, meta_path,
    )


if __name__ == "__main__":
    main()
