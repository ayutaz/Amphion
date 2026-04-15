"""Tests for CoMelSinger Dataset, Sampler, and collate_fn (M2-07 to M2-10)."""
from __future__ import annotations

import pytest
import torch
from pathlib import Path
from typing import List

from models.tts.comelsinger.dataset import (
    CoMelSingerDataset,
    BalancedSpeakerSampler,
    comelsinger_collate_fn,
    prepare_batch_for_svt,
)
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(
    T: int = 50,
    n_phones: int = 10,
    n_notes: int = 5,
    speaker_id: int = 0,
    acoustic_shape: str = "12T",  # "12T" or "T12"
) -> dict:
    """Create a minimal sample dict matching the .pt file schema."""
    if acoustic_shape == "12T":
        acoustic = torch.randint(0, 1024, (12, T), dtype=torch.long)
    else:
        acoustic = torch.randint(0, 1024, (T, 12), dtype=torch.long)
    return {
        "acoustic_tokens": acoustic,
        "semantic_tokens": torch.randint(0, 1000, (T,), dtype=torch.long),
        "pitch_tokens": torch.randint(0, 128, (T,), dtype=torch.long),
        "phone_ids": torch.randint(0, 50, (n_phones,), dtype=torch.long),
        "note_durations": torch.randint(1, 10, (n_notes,), dtype=torch.long),
        "note_pitches": torch.randint(40, 80, (n_notes,), dtype=torch.long),
        "attention_mask": torch.ones(T, dtype=torch.long),
        "speaker_id": torch.tensor(speaker_id, dtype=torch.long),
    }


def _write_samples(tmp_path: Path, samples: List[dict], subdir: str = "tokens") -> Path:
    """Save sample dicts as .pt files under tmp_path/subdir/."""
    out_dir = tmp_path / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(samples):
        torch.save(sample, out_dir / f"sample_{i:04d}.pt")
    return tmp_path


# ---------------------------------------------------------------------------
# TestCoMelSingerDataset
# ---------------------------------------------------------------------------


class TestCoMelSingerDataset:
    def test_len(self, tmp_path):
        """Dataset.__len__ matches number of .pt files created."""
        samples = [_make_sample(T=50, speaker_id=0) for _ in range(6)]
        _write_samples(tmp_path, samples)
        ds = CoMelSingerDataset(tmp_path)
        assert len(ds) == 6

    def test_getitem_shapes(self, tmp_path):
        """__getitem__ returns tensors with expected shapes."""
        T = 60
        _write_samples(tmp_path, [_make_sample(T=T, n_phones=8, n_notes=4, speaker_id=1)])
        ds = CoMelSingerDataset(tmp_path)
        item = ds[0]

        assert item["acoustic_tokens"].shape == (12, T)
        assert item["semantic_tokens"].shape == (T,)
        assert item["pitch_tokens"].shape == (T,)
        assert item["phone_ids"].shape == (8,)
        assert item["note_durations"].shape == (4,)
        assert item["note_pitches"].shape == (4,)
        assert item["attention_mask"].shape == (T,)

    def test_transposition_T12_to_12T(self, tmp_path):
        """(T, 12) acoustic_tokens is transposed to (12, T) by __getitem__."""
        T = 40
        sample = _make_sample(T=T, acoustic_shape="T12", speaker_id=0)
        # Verify the saved shape is (T, 12)
        assert sample["acoustic_tokens"].shape == (T, 12)
        _write_samples(tmp_path, [sample])
        ds = CoMelSingerDataset(tmp_path)
        item = ds[0]
        assert item["acoustic_tokens"].shape == (12, T)

    def test_speaker_id_is_int(self, tmp_path):
        """speaker_id in returned dict is a Python int."""
        _write_samples(tmp_path, [_make_sample(speaker_id=3)])
        ds = CoMelSingerDataset(tmp_path)
        item = ds[0]
        assert isinstance(item["speaker_id"], int)
        assert item["speaker_id"] == 3

    def test_empty_directory_raises_value_error(self, tmp_path):
        """Passing a directory with no .pt files raises ValueError."""
        (tmp_path / "tokens").mkdir()
        with pytest.raises(ValueError, match="No .pt files found"):
            CoMelSingerDataset(tmp_path)

    def test_fallback_to_data_dir(self, tmp_path):
        """When no tokens/ subdir exists, .pt files directly in data_dir are used."""
        sample = _make_sample(T=30, speaker_id=0)
        # Write directly under tmp_path (no tokens/ subdir)
        torch.save(sample, tmp_path / "sample_0000.pt")
        ds = CoMelSingerDataset(tmp_path)
        assert len(ds) == 1

    def test_speaker_to_indices_built(self, tmp_path):
        """speaker_to_indices maps each speaker ID to correct sample indices."""
        samples = [
            _make_sample(T=30, speaker_id=0),
            _make_sample(T=30, speaker_id=1),
            _make_sample(T=30, speaker_id=0),
        ]
        _write_samples(tmp_path, samples)
        ds = CoMelSingerDataset(tmp_path)
        # speaker 0 should have indices 0 and 2 (sorted by filename)
        assert 0 in ds.speaker_to_indices
        assert 1 in ds.speaker_to_indices
        assert len(ds.speaker_to_indices[0]) == 2
        assert len(ds.speaker_to_indices[1]) == 1


# ---------------------------------------------------------------------------
# TestBalancedSpeakerSampler
# ---------------------------------------------------------------------------


class TestBalancedSpeakerSampler:
    def _make_speaker_ids(self, n_per_speaker: int = 20, n_speakers: int = 4) -> List[int]:
        ids = []
        for spk in range(n_speakers):
            ids.extend([spk] * n_per_speaker)
        return ids

    def test_batch_size_matches(self):
        """Every yielded batch has exactly batch_size elements."""
        speaker_ids = self._make_speaker_ids(n_per_speaker=20, n_speakers=4)
        sampler = BalancedSpeakerSampler(speaker_ids, batch_size=8, k_s=2, seed=0)
        for batch in sampler:
            assert len(batch) == 8

    def test_head_contains_same_speaker_pair(self):
        """First k_s indices in each batch include at least one same-speaker pair."""
        speaker_ids = self._make_speaker_ids(n_per_speaker=20, n_speakers=4)
        k_s = 4
        sampler = BalancedSpeakerSampler(speaker_ids, batch_size=8, k_s=k_s, seed=42)

        for batch in sampler:
            head = batch[:k_s]
            head_speakers = [speaker_ids[i] for i in head]
            # At least two indices share the same speaker
            from collections import Counter
            counts = Counter(head_speakers)
            assert max(counts.values()) >= 2, (
                f"No same-speaker pair in head: {head_speakers}"
            )

    def test_epoch_reproducibility(self):
        """Same seed produces identical batch sequence on two iterations."""
        speaker_ids = self._make_speaker_ids(n_per_speaker=16, n_speakers=4)
        sampler = BalancedSpeakerSampler(speaker_ids, batch_size=8, k_s=2, seed=7)
        run1 = list(sampler)
        # Re-iterate (same object, same seed)
        run2 = list(sampler)
        assert run1 == run2

    def test_different_seeds_different_batches(self):
        """Different seeds produce different batch orders."""
        speaker_ids = self._make_speaker_ids(n_per_speaker=16, n_speakers=4)
        s1 = BalancedSpeakerSampler(speaker_ids, batch_size=8, k_s=2, seed=1)
        s2 = BalancedSpeakerSampler(speaker_ids, batch_size=8, k_s=2, seed=2)
        batches1 = list(s1)
        batches2 = list(s2)
        # Very unlikely to be identical with different seeds
        assert batches1 != batches2

    def test_no_eligible_speakers_raises(self):
        """All speakers with only 1 sample raise ValueError."""
        speaker_ids = [0, 1, 2, 3]  # each speaker appears exactly once
        with pytest.raises(ValueError, match="No speaker has >= 2 samples"):
            BalancedSpeakerSampler(speaker_ids, batch_size=2, k_s=1)

    def test_len_equals_floor_division(self):
        """__len__ == len(speaker_ids) // batch_size."""
        speaker_ids = self._make_speaker_ids(n_per_speaker=20, n_speakers=4)
        sampler = BalancedSpeakerSampler(speaker_ids, batch_size=8, k_s=2, seed=0)
        assert len(sampler) == len(speaker_ids) // 8


# ---------------------------------------------------------------------------
# TestCollateFn
# ---------------------------------------------------------------------------


class TestCollateFn:
    def _make_items(self, lengths: List[int], speaker_ids: List[int] = None) -> List[dict]:
        """Create raw items (as returned by CoMelSingerDataset.__getitem__)."""
        if speaker_ids is None:
            speaker_ids = [0] * len(lengths)
        items = []
        for T, sid in zip(lengths, speaker_ids):
            items.append({
                "acoustic_tokens": torch.randint(0, 1024, (12, T), dtype=torch.long),
                "semantic_tokens": torch.randint(0, 1000, (T,), dtype=torch.long),
                "pitch_tokens": torch.randint(0, 128, (T,), dtype=torch.long),
                "phone_ids": torch.randint(0, 50, (5,), dtype=torch.long),
                "note_durations": torch.randint(1, 10, (3,), dtype=torch.long),
                "note_pitches": torch.randint(40, 80, (3,), dtype=torch.long),
                "attention_mask": torch.ones(T, dtype=torch.long),
                "speaker_id": sid,
            })
        return items

    def test_output_shape_acoustic(self):
        """acoustic_tokens has shape (B, 12, T_max)."""
        lengths = [30, 50, 40]
        batch = comelsinger_collate_fn(self._make_items(lengths))
        assert batch["acoustic_tokens"].shape == (3, 12, 50)

    def test_output_shape_semantic_and_pitch(self):
        """semantic_tokens and pitch_tokens have shape (B, T_max)."""
        lengths = [20, 35]
        batch = comelsinger_collate_fn(self._make_items(lengths))
        assert batch["semantic_tokens"].shape == (2, 35)
        assert batch["pitch_tokens"].shape == (2, 35)

    def test_padding_zeros_outside_original_length(self):
        """Padded positions in acoustic_tokens are zero."""
        items = self._make_items([10, 20])
        batch = comelsinger_collate_fn(items)
        # Sample 0 has T=10, padded from index 10 to 20
        assert (batch["acoustic_tokens"][0, :, 10:] == 0).all()

    def test_attention_mask_shape_and_values(self):
        """attention_mask shape is (B, T_max); 1 for valid, 0 for padded."""
        items = self._make_items([10, 25])
        batch = comelsinger_collate_fn(items)
        assert batch["attention_mask"].shape == (2, 25)
        # First sample: valid positions 0..9 == 1, padded 10..24 == 0
        assert (batch["attention_mask"][0, :10] == 1).all()
        assert (batch["attention_mask"][0, 10:] == 0).all()
        # Second sample: all valid
        assert (batch["attention_mask"][1, :] == 1).all()

    def test_speaker_id_shape_and_values(self):
        """speaker_id is (B,) long tensor with correct values."""
        items = self._make_items([20, 20], speaker_ids=[3, 7])
        batch = comelsinger_collate_fn(items)
        assert batch["speaker_id"].shape == (2,)
        assert batch["speaker_id"][0].item() == 3
        assert batch["speaker_id"][1].item() == 7

    def test_ragged_note_durations(self):
        """note_durations is a list of tensors (ragged, not padded)."""
        items = [
            {
                "acoustic_tokens": torch.zeros(12, 30, dtype=torch.long),
                "semantic_tokens": torch.zeros(30, dtype=torch.long),
                "pitch_tokens": torch.zeros(30, dtype=torch.long),
                "attention_mask": torch.ones(30, dtype=torch.long),
                "speaker_id": 0,
                "note_durations": torch.tensor([5, 3, 2], dtype=torch.long),
                "note_pitches": torch.tensor([60, 62, 64], dtype=torch.long),
                "phone_ids": torch.tensor([1, 2], dtype=torch.long),
            },
            {
                "acoustic_tokens": torch.zeros(12, 40, dtype=torch.long),
                "semantic_tokens": torch.zeros(40, dtype=torch.long),
                "pitch_tokens": torch.zeros(40, dtype=torch.long),
                "attention_mask": torch.ones(40, dtype=torch.long),
                "speaker_id": 1,
                "note_durations": torch.tensor([4, 6], dtype=torch.long),
                "note_pitches": torch.tensor([67, 69], dtype=torch.long),
                "phone_ids": torch.tensor([3, 4, 5], dtype=torch.long),
            },
        ]
        batch = comelsinger_collate_fn(items)
        assert isinstance(batch["note_durations"], list)
        assert batch["note_durations"][0].tolist() == [5, 3, 2]
        assert batch["note_durations"][1].tolist() == [4, 6]

    def test_single_item_batch(self):
        """Collate with B=1 works without errors."""
        items = self._make_items([15])
        batch = comelsinger_collate_fn(items)
        assert batch["acoustic_tokens"].shape == (1, 12, 15)

    def test_equal_length_batch(self):
        """All samples with same T should produce T_max == T."""
        items = self._make_items([20, 20, 20])
        batch = comelsinger_collate_fn(items)
        assert batch["acoustic_tokens"].shape == (3, 12, 20)


# ---------------------------------------------------------------------------
# TestDataLoaderIntegration
# ---------------------------------------------------------------------------


class TestDataLoaderIntegration:
    def test_dataloader_one_batch(self, tmp_path):
        """DataLoader with BalancedSpeakerSampler and collate_fn yields one valid batch."""
        # Create 32 samples: 4 speakers × 8 samples each
        samples = []
        for spk in range(4):
            for _ in range(8):
                T = torch.randint(30, 60, (1,)).item()
                samples.append(_make_sample(T=T, speaker_id=spk))
        _write_samples(tmp_path, samples)

        ds = CoMelSingerDataset(tmp_path)
        sampler = BalancedSpeakerSampler(
            ds.speaker_ids,
            batch_size=8,
            k_s=2,
            seed=0,
        )
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=comelsinger_collate_fn,
            num_workers=0,
        )

        batch = next(iter(loader))

        assert "acoustic_tokens" in batch
        assert "semantic_tokens" in batch
        assert "pitch_tokens" in batch
        assert "attention_mask" in batch
        assert "speaker_id" in batch
        assert "note_durations" in batch
        assert "note_pitches" in batch
        assert "phone_ids" in batch

        B = batch["acoustic_tokens"].shape[0]
        assert B == 8
        assert batch["acoustic_tokens"].ndim == 3
        assert batch["acoustic_tokens"].shape[1] == 12
        assert batch["semantic_tokens"].shape == (B, batch["semantic_tokens"].shape[-1])
        assert batch["speaker_id"].shape == (B,)
        assert isinstance(batch["note_durations"], list)
        assert len(batch["note_durations"]) == B

    def test_dataloader_all_batches_valid(self, tmp_path):
        """All batches from DataLoader have expected structure."""
        n_per_speaker = 12
        n_speakers = 4
        samples = []
        for spk in range(n_speakers):
            for _ in range(n_per_speaker):
                T = 40
                samples.append(_make_sample(T=T, speaker_id=spk))
        _write_samples(tmp_path, samples)

        ds = CoMelSingerDataset(tmp_path)
        sampler = BalancedSpeakerSampler(
            ds.speaker_ids,
            batch_size=8,
            k_s=2,
            seed=1,
        )
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=comelsinger_collate_fn,
            num_workers=0,
        )

        for batch in loader:
            B = batch["acoustic_tokens"].shape[0]
            assert B == 8
            T_max = batch["acoustic_tokens"].shape[-1]
            assert batch["semantic_tokens"].shape == (B, T_max)
            assert batch["pitch_tokens"].shape == (B, T_max)
            assert batch["attention_mask"].shape == (B, T_max)
            assert batch["speaker_id"].shape == (B,)
            assert len(batch["note_durations"]) == B


# ---------------------------------------------------------------------------
# TestManifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_manifest_created_on_first_load(self, tmp_path):
        """First load creates manifest.json automatically."""
        samples = [_make_sample(T=30, speaker_id=i % 2) for i in range(4)]
        _write_samples(tmp_path, samples)
        manifest_file = tmp_path / "manifest.json"
        assert not manifest_file.exists()
        ds = CoMelSingerDataset(tmp_path)
        assert manifest_file.exists()
        assert len(ds) == 4

    def test_manifest_fast_path(self, tmp_path):
        """Second load reads manifest.json without scanning .pt files."""
        import json
        samples = [_make_sample(T=30, speaker_id=i % 2) for i in range(4)]
        _write_samples(tmp_path, samples)
        # First load: creates manifest
        ds1 = CoMelSingerDataset(tmp_path)
        # Second load: uses manifest (fast path)
        ds2 = CoMelSingerDataset(tmp_path)
        assert len(ds2) == len(ds1)
        assert ds2.speaker_ids == ds1.speaker_ids

    def test_custom_manifest_path(self, tmp_path):
        """Custom manifest_path creates manifest at the specified location."""
        samples = [_make_sample(T=30, speaker_id=0) for _ in range(3)]
        _write_samples(tmp_path, samples)
        custom_path = tmp_path / "custom_manifest.json"
        ds = CoMelSingerDataset(tmp_path, manifest_path=custom_path)
        assert custom_path.exists()
        assert len(ds) == 3


# ---------------------------------------------------------------------------
# TestPrepareBatchForSvt
# ---------------------------------------------------------------------------


class TestPrepareBatchForSvt:
    def test_permute_shape(self):
        """prepare_batch_for_svt transposes acoustic_tokens from (B,12,T) to (B,T,12)."""
        items = [
            {
                "acoustic_tokens": torch.randint(0, 1024, (12, 30), dtype=torch.long),
                "semantic_tokens": torch.randint(0, 1000, (30,), dtype=torch.long),
                "pitch_tokens": torch.randint(0, 128, (30,), dtype=torch.long),
                "attention_mask": torch.ones(30, dtype=torch.long),
                "speaker_id": 0,
                "note_durations": torch.tensor([10, 10, 10], dtype=torch.long),
                "note_pitches": torch.tensor([60, 62, 64], dtype=torch.long),
                "phone_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            }
        ]
        batch = comelsinger_collate_fn(items)
        assert batch["acoustic_tokens"].shape == (1, 12, 30)
        svt_batch = prepare_batch_for_svt(batch)
        assert svt_batch["acoustic_tokens"].shape == (1, 30, 12)

    def test_does_not_mutate_original(self):
        """prepare_batch_for_svt returns a new dict, original is unchanged."""
        items = [
            {
                "acoustic_tokens": torch.randint(0, 1024, (12, 20), dtype=torch.long),
                "semantic_tokens": torch.randint(0, 1000, (20,), dtype=torch.long),
                "pitch_tokens": torch.randint(0, 128, (20,), dtype=torch.long),
                "attention_mask": torch.ones(20, dtype=torch.long),
                "speaker_id": 0,
                "note_durations": torch.tensor([10, 10], dtype=torch.long),
                "note_pitches": torch.tensor([60, 62], dtype=torch.long),
                "phone_ids": torch.tensor([1], dtype=torch.long),
            }
        ]
        batch = comelsinger_collate_fn(items)
        original_shape = batch["acoustic_tokens"].shape
        _ = prepare_batch_for_svt(batch)
        assert batch["acoustic_tokens"].shape == original_shape


# ---------------------------------------------------------------------------
# TestE2EPipeline (T1-T3)
# ---------------------------------------------------------------------------


class TestE2EPipeline:
    """End-to-end tests: Dataset -> collate -> SVT/S2A."""

    def test_collate_to_svt(self):
        """collate output -> permute -> SVTModule.forward() produces valid logits."""
        from models.tts.comelsinger.svt_module import SVTModule

        B, T = 2, 40
        items = [
            {
                "acoustic_tokens": torch.randint(0, 1024, (12, T), dtype=torch.long),
                "semantic_tokens": torch.randint(0, 1000, (T,), dtype=torch.long),
                "pitch_tokens": torch.randint(0, 128, (T,), dtype=torch.long),
                "attention_mask": torch.ones(T, dtype=torch.long),
                "speaker_id": i,
                "note_durations": torch.tensor([T // 4] * 4, dtype=torch.long),
                "note_pitches": torch.randint(40, 80, (4,), dtype=torch.long),
                "phone_ids": torch.randint(0, 50, (5,), dtype=torch.long),
            }
            for i in range(B)
        ]

        batch = comelsinger_collate_fn(items)
        svt_batch = prepare_batch_for_svt(batch)

        model = SVTModule()
        model.eval()
        with torch.no_grad():
            out = model(
                svt_batch["acoustic_tokens"],
                attention_mask=svt_batch["attention_mask"],
            )

        assert out["logits"].shape == (B, T, 129)
        assert torch.isfinite(out["logits"]).all()

    def test_collate_to_s2a(self):
        """collate output -> permute -> CoMelSinger_S2A.forward() runs."""
        from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A

        B, T = 2, 50
        items = [
            {
                "acoustic_tokens": torch.randint(0, 1024, (12, T), dtype=torch.long),
                "semantic_tokens": torch.randint(0, 1000, (T,), dtype=torch.long),
                "pitch_tokens": torch.randint(0, 128, (T,), dtype=torch.long),
                "attention_mask": torch.ones(T, dtype=torch.long),
                "speaker_id": i,
                "note_durations": torch.tensor([T // 5] * 5, dtype=torch.long),
                "note_pitches": torch.randint(40, 80, (5,), dtype=torch.long),
                "phone_ids": torch.randint(0, 50, (5,), dtype=torch.long),
            }
            for i in range(B)
        ]

        batch = comelsinger_collate_fn(items)
        svt_batch = prepare_batch_for_svt(batch)

        model = CoMelSinger_S2A()
        result = model(
            x0=svt_batch["acoustic_tokens"],
            x_mask=svt_batch["attention_mask"],
            cond_code=batch["semantic_tokens"],
            pitch_tokens=batch["pitch_tokens"],
        )
        # forward returns tuple of 6 elements
        assert len(result) == 6
        logits = result[0]
        assert torch.isfinite(logits).all()

    def test_svt_frozen_with_s2a(self):
        """S2A forward + frozen SVT -> compute_total_loss end-to-end."""
        from models.tts.comelsinger.svt_module import SVTModule
        from models.tts.comelsinger.losses import compute_svt_loss, compute_total_loss

        B, T = 2, 40
        items = [
            {
                "acoustic_tokens": torch.randint(0, 1024, (12, T), dtype=torch.long),
                "semantic_tokens": torch.randint(0, 1000, (T,), dtype=torch.long),
                "pitch_tokens": torch.randint(0, 128, (T,), dtype=torch.long),
                "attention_mask": torch.ones(T, dtype=torch.long),
                "speaker_id": i,
                "note_durations": torch.tensor([T // 4] * 4, dtype=torch.long),
                "note_pitches": torch.randint(40, 80, (4,), dtype=torch.long),
                "phone_ids": torch.randint(0, 50, (5,), dtype=torch.long),
            }
            for i in range(B)
        ]

        batch = comelsinger_collate_fn(items)
        svt_batch = prepare_batch_for_svt(batch)

        # Frozen SVT
        svt = SVTModule()
        svt.freeze()
        with torch.no_grad():
            svt_out = svt(
                svt_batch["acoustic_tokens"],
                attention_mask=svt_batch["attention_mask"],
            )

        # Build frame alignment / pitch note labels from batch
        frame_alignment = [d.tolist() for d in batch["note_durations"]]
        pitch_note_labels = [p.tolist() for p in batch["note_pitches"]]

        svt_loss, svt_parts = compute_svt_loss(
            pitch_logits=svt_out["logits"],
            target_pitch_tokens=batch["pitch_tokens"],
            frame_alignment=frame_alignment,
            pitch_note_labels=pitch_note_labels,
            attention_mask=batch["attention_mask"],
        )

        # Dummy losses for SCL, FCL, mask (would come from S2A in real pipeline)
        l_scl = torch.tensor(0.5, requires_grad=True)
        l_fcl = torch.tensor(0.3, requires_grad=True)
        l_mask = torch.tensor(1.0, requires_grad=True)

        total, parts = compute_total_loss(l_scl, l_fcl, svt_loss.detach(), l_mask)
        assert torch.isfinite(total)
        total.backward()

        # SVT params should still have no grad
        for name, p in svt.named_parameters():
            assert p.grad is None, f"Frozen SVT param {name} received grad"

        # Dummy losses should have grads
        assert l_scl.grad is not None
        assert l_fcl.grad is not None
        assert l_mask.grad is not None
