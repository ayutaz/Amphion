"""CoMelSinger data preprocessing pipeline.

Extracts acoustic tokens, semantic tokens, pitch tokens, and phone IDs
from audio files and saves them as .pt files.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# --------------------------------------------------------------------------
# M1-12: Acoustic token extraction
# --------------------------------------------------------------------------

def extract_acoustic_tokens(
    speech: torch.Tensor,
    codec_encoder,
    codec_decoder,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Extract RVQ acoustic tokens from speech waveform.

    Args:
        speech: (T,) waveform at 24kHz, float32
        codec_encoder: Amphion CodecEncoder instance
        codec_decoder: Amphion CodecDecoder instance (for quantizer)
        device: computation device

    Returns:
        acoustic_tokens: (T_a, 12) long, values in [0, 1023]
    """
    speech = speech.to(device)
    with torch.no_grad():
        vq_emb = codec_encoder(speech.unsqueeze(0).unsqueeze(0))
        _, vq, _, _, _ = codec_decoder.quantizer(vq_emb)
        # vq: (num_codebooks, B, T_a) -> (B, T_a, num_codebooks) -> squeeze
        acoustic_tokens = vq.permute(1, 2, 0).squeeze(0).long()
    return acoustic_tokens


# --------------------------------------------------------------------------
# M1-13: Semantic token extraction
# --------------------------------------------------------------------------

def extract_semantic_tokens(
    speech: torch.Tensor,
    w2v_bert_model,
    semantic_codec,
    semantic_mean: torch.Tensor,
    semantic_std: torch.Tensor,
    target_len: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Extract semantic tokens using w2v-bert-2.0 + RepCodec.

    Args:
        speech: (T,) waveform at 16kHz, float32
        w2v_bert_model: Wav2Vec2BertModel instance
        semantic_codec: RepCodec quantizer
        semantic_mean, semantic_std: z-score normalization stats, shape (D,)
        target_len: T_a (75Hz target frame count)
        device: computation device

    Returns:
        semantic_tokens: (T_a,) long
    """
    speech = speech.to(device)
    with torch.no_grad():
        outputs = w2v_bert_model(
            speech.unsqueeze(0), output_hidden_states=True
        )
        feat = outputs.hidden_states[17].squeeze(0)  # (T_50, D)

        # z-score normalization
        feat = (feat - semantic_mean.to(device)) / (semantic_std.to(device) + 1e-8)

        # 50Hz -> 75Hz nearest-neighbor upsample
        feat = feat.unsqueeze(0).permute(0, 2, 1)  # (1, D, T_50)
        feat = torch.nn.functional.interpolate(
            feat, size=target_len, mode="nearest"
        )
        feat = feat.permute(0, 2, 1).squeeze(0)  # (T_a, D)

        # VQ encode
        semantic_tokens, _ = semantic_codec.quantize(feat.unsqueeze(0))
        semantic_tokens = semantic_tokens.squeeze(0)  # (T_a,)
    return semantic_tokens


# --------------------------------------------------------------------------
# M1-14: Pitch token extraction
# --------------------------------------------------------------------------

def extract_pitch_tokens(
    speech: torch.Tensor,
    pitch_tokenizer,
    target_len: int,
    sr: int = 24000,
    frame_period: float = 5.0,
    f0_floor: float = 65.0,
    f0_ceil: float = 1047.0,
) -> torch.Tensor:
    """Extract pitch tokens using pyworld + PitchTokenizer.

    Args:
        speech: (T,) waveform at 24kHz, float32
        pitch_tokenizer: PitchTokenizer instance
        target_len: T_a (75Hz target frame count)
        sr: sample rate (default 24000)
        frame_period: DIO frame period in ms (default 5.0)
        f0_floor: minimum F0 in Hz (default 65.0)
        f0_ceil: maximum F0 in Hz (default 1047.0)

    Returns:
        pitch_tokens: (T_a,) long, values in [0, 128]
    """
    import pyworld as pw

    speech_np = speech.numpy().astype(np.float64)
    f0, t = pw.dio(
        speech_np, sr,
        f0_floor=f0_floor, f0_ceil=f0_ceil,
        frame_period=frame_period,
    )
    f0 = pw.stonemask(speech_np, f0, t, sr)
    pitch_tokens = pitch_tokenizer.quantize_f0(f0, target_len=target_len)
    return pitch_tokens


# --------------------------------------------------------------------------
# M1-15: Phone ID extraction
# --------------------------------------------------------------------------

def extract_phone_ids(
    text: str,
    phone2id: dict,
    language: str = "zh",
) -> torch.Tensor:
    """Extract phone IDs from text using pypinyin + G2P.

    Args:
        text: lyrics text (Chinese)
        phone2id: phone-to-ID mapping dict
        language: language code

    Returns:
        phone_ids: (T_ph,) long
    """
    from pypinyin import lazy_pinyin, Style

    pinyins = lazy_pinyin(text, style=Style.TONE3)
    # Flatten pinyin to character-level phones
    phones = []
    for py in pinyins:
        phones.extend(list(py))

    ids = [phone2id.get(p, phone2id.get("<unk>", 0)) for p in phones]
    return torch.tensor(ids, dtype=torch.long)


# --------------------------------------------------------------------------
# M1-16: Save preprocessed data
# --------------------------------------------------------------------------

def save_preprocessed(
    output_path: str | Path,
    acoustic_tokens: torch.Tensor,
    semantic_tokens: torch.Tensor,
    pitch_tokens: torch.Tensor,
    phone_ids: torch.Tensor,
    note_durations: torch.Tensor,
    note_pitches: torch.Tensor,
    attention_mask: torch.Tensor,
    speaker_id: int,
) -> None:
    """Save all preprocessed features to a .pt file.

    acoustic_tokens is transposed from (T_a, 12) to (12, T_a) for DataLoader.

    Raises:
        ValueError: if length consistency check fails
    """
    T_a = acoustic_tokens.shape[0]
    for name, tensor in [
        ("semantic_tokens", semantic_tokens),
        ("pitch_tokens", pitch_tokens),
        ("attention_mask", attention_mask),
    ]:
        if tensor.shape[0] != T_a:
            raise ValueError(f"{name} length {tensor.shape[0]} != T_a {T_a}")

    data = {
        "acoustic_tokens": acoustic_tokens.permute(1, 0).contiguous(),  # (12, T_a)
        "semantic_tokens": semantic_tokens,
        "pitch_tokens": pitch_tokens,
        "phone_ids": phone_ids,
        "note_durations": note_durations,
        "note_pitches": note_pitches,
        "attention_mask": attention_mask,
        "speaker_id": torch.tensor(speaker_id, dtype=torch.long),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)


def load_preprocessed(path: str | Path) -> dict:
    """Load preprocessed .pt file."""
    return torch.load(path, map_location="cpu", weights_only=True)
