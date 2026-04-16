"""CoMelSinger Dataset, Sampler, and collate function."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset, Sampler, DataLoader


class CoMelSingerDataset(Dataset):
    """Dataset that loads preprocessed .pt files from tokens/ directory.

    Each .pt file contains:
        acoustic_tokens: (12, T) or (T, 12) long
        semantic_tokens: (T,) long
        pitch_tokens: (T,) long
        phone_ids: (T_ph,) long
        note_durations: (S,) long
        note_pitches: (S,) long
        attention_mask: (T,) long
        speaker_id: int/long scalar
    """

    def __init__(self, data_dir: str | Path, manifest_path: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir)
        token_dir = self.data_dir / "tokens"
        if not token_dir.exists():
            token_dir = self.data_dir  # fallback: .pt files directly in data_dir

        manifest_file = self.data_dir / "manifest.json" if manifest_path is None else Path(manifest_path)

        if manifest_file.exists():
            # Fast path: read pre-computed manifest
            with open(manifest_file) as f:
                manifest = json.load(f)
            self.samples = [Path(m["path"]) for m in manifest]
            self.speaker_ids: List[int] = [m["speaker_id"] for m in manifest]
        else:
            # Slow path: scan .pt files and build manifest
            self.samples = sorted(token_dir.glob("*.pt"))
            if not self.samples:
                raise ValueError(f"No .pt files found in {token_dir}")
            self.speaker_ids = []
            for path in self.samples:
                data = torch.load(path, map_location="cpu", weights_only=True)
                self.speaker_ids.append(int(data["speaker_id"]))
            # Save manifest for next time
            manifest = [{"path": str(p), "speaker_id": s} for p, s in zip(self.samples, self.speaker_ids)]
            with open(manifest_file, "w") as f:
                json.dump(manifest, f)

        if not self.samples:
            raise ValueError(f"No .pt files found in {token_dir}")

        # Build speaker index for BalancedSpeakerSampler
        self.speaker_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, sid in enumerate(self.speaker_ids):
            self.speaker_to_indices[sid].append(idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int | str]:
        data = torch.load(self.samples[idx], map_location="cpu", weights_only=True)
        acoustic = data["acoustic_tokens"]
        # Ensure shape is (12, T) - first dim is codebooks
        if acoustic.ndim == 2 and acoustic.shape[-1] == 12:
            acoustic = acoustic.T
        return {
            "acoustic_tokens": acoustic,  # (12, T)
            "semantic_tokens": data["semantic_tokens"],  # (T,)
            "pitch_tokens": data["pitch_tokens"],  # (T,)
            "phone_ids": data.get("phone_ids", torch.empty(0, dtype=torch.long)),
            "note_durations": data.get("note_durations", torch.empty(0, dtype=torch.long)),
            "note_pitches": data.get("note_pitches", torch.empty(0, dtype=torch.long)),
            "attention_mask": data.get("attention_mask", torch.ones(acoustic.shape[-1], dtype=torch.long)),
            "speaker_id": int(data["speaker_id"]),
        }


class BalancedSpeakerSampler(Sampler):
    """Sampler that ensures first k_s samples in each batch share speaker pairs.

    For SCL (Sequence-level Contrastive Learning), the first k_s samples
    must contain at least one pair of same-speaker samples.

    Args:
        speaker_ids: list of speaker IDs, one per dataset sample
        batch_size: total batch size K (default 32)
        k_s: number of SCL samples at batch head (default 8)
        seed: random seed for reproducibility
    """

    def __init__(
        self,
        speaker_ids: List[int],
        batch_size: int = 32,
        k_s: int = 8,
        seed: int = 42,
    ) -> None:
        self.speaker_ids = speaker_ids
        self.batch_size = batch_size
        self.k_s = k_s
        self.seed = seed

        self.speaker_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, sid in enumerate(speaker_ids):
            self.speaker_to_indices[sid].append(idx)

        # Only speakers with >= 2 samples can form SCL pairs
        self.eligible_speakers = [
            sid for sid, indices in self.speaker_to_indices.items()
            if len(indices) >= 2
        ]
        if not self.eligible_speakers:
            raise ValueError("No speaker has >= 2 samples for SCL pairs")

    def set_epoch(self, epoch: int) -> None:
        """Update seed for epoch-based shuffling (DDP compatible)."""
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + getattr(self, "epoch", 0))
        all_indices = list(range(len(self.speaker_ids)))
        rng.shuffle(all_indices)

        # Generate batches — use set for O(1) membership checks
        remaining = set(all_indices)
        while len(remaining) >= self.batch_size:
            batch = []
            batch_set = set()

            # First k_s: pick a speaker with >=2 samples, add pair + fill rest
            speaker = rng.choice(self.eligible_speakers)
            pool = list(self.speaker_to_indices[speaker])
            rng.shuffle(pool)
            pair = pool[:2]
            batch.extend(pair)
            batch_set.update(pair)
            # Fill remaining k_s slots from remaining indices
            fill_needed = self.k_s - len(batch)
            fill_candidates = list(remaining - batch_set)
            rng.shuffle(fill_candidates)
            batch.extend(fill_candidates[:fill_needed])
            batch_set.update(fill_candidates[:fill_needed])

            # Rest of batch (k_s to batch_size)
            rest_candidates = list(remaining - batch_set)
            rng.shuffle(rest_candidates)
            batch.extend(rest_candidates[:self.batch_size - len(batch)])
            batch_set.update(rest_candidates[:self.batch_size - len(batch)])

            # Remove used indices
            remaining -= batch_set

            yield batch

    def __len__(self) -> int:
        return len(self.speaker_ids) // self.batch_size


def comelsinger_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor | List]:
    """Collate variable-length samples with zero-padding.

    Returns:
        acoustic_tokens: (B, 12, T_max) long
        semantic_tokens: (B, T_max) long
        pitch_tokens: (B, T_max) long
        attention_mask: (B, T_max) long
        speaker_id: (B,) long
        note_durations: List[Tensor] (ragged, not padded)
        note_pitches: List[Tensor] (ragged, not padded)
        phone_ids: List[Tensor] (ragged, not padded)

    NOTE: acoustic_tokens is returned as (B, 12, T_max). Downstream modules
    (SVTModule, CoMelSinger_S2A) expect (B, T, 12). Use
    ``prepare_batch_for_svt(batch)`` or ``batch["acoustic_tokens"].permute(0, 2, 1)``
    before passing to those modules.
    """
    # Find T_max
    T_max = max(item["acoustic_tokens"].shape[-1] for item in batch)
    B = len(batch)

    acoustic = torch.zeros(B, 12, T_max, dtype=torch.long)
    semantic = torch.zeros(B, T_max, dtype=torch.long)
    pitch = torch.zeros(B, T_max, dtype=torch.long)
    mask = torch.zeros(B, T_max, dtype=torch.long)
    speaker = torch.zeros(B, dtype=torch.long)

    note_durations = []
    note_pitches = []
    phone_ids = []

    for i, item in enumerate(batch):
        T = item["acoustic_tokens"].shape[-1]
        acoustic[i, :, :T] = item["acoustic_tokens"]
        semantic[i, :T] = item["semantic_tokens"]
        pitch[i, :T] = item["pitch_tokens"]
        mask[i, :T] = item.get("attention_mask", torch.ones(T, dtype=torch.long))[:T]
        speaker[i] = item["speaker_id"]
        note_durations.append(item.get("note_durations", torch.empty(0, dtype=torch.long)))
        note_pitches.append(item.get("note_pitches", torch.empty(0, dtype=torch.long)))
        phone_ids.append(item.get("phone_ids", torch.empty(0, dtype=torch.long)))

    return {
        "acoustic_tokens": acoustic,  # (B, 12, T_max)
        "semantic_tokens": semantic,  # (B, T_max)
        "pitch_tokens": pitch,  # (B, T_max)
        "attention_mask": mask,  # (B, T_max)
        "speaker_id": speaker,  # (B,)
        "note_durations": note_durations,  # List[Tensor]
        "note_pitches": note_pitches,  # List[Tensor]
        "phone_ids": phone_ids,  # List[Tensor]
    }


def prepare_batch_for_svt(batch: Dict) -> Dict:
    """Convert collate_fn output to SVTModule / CoMelSinger_S2A input format.

    acoustic_tokens: (B, 12, T) -> (B, T, 12)

    Returns a shallow copy of the batch dict with the transposed tensor.
    """
    batch = dict(batch)  # shallow copy
    batch["acoustic_tokens"] = batch["acoustic_tokens"].permute(0, 2, 1)
    return batch
