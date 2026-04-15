"""SVT (Singing Voice Transcription) Training Script.

Trains SVTModule independently on acoustic tokens -> pitch token prediction.
Paper Section III-C, IV-B.

Usage:
    uv run python models/tts/comelsinger/train_svt.py --config configs/comelsinger/svt_train.yaml
    uv run python models/tts/comelsinger/train_svt.py --config configs/comelsinger/svt_train.yaml --dry-run
    uv run python models/tts/comelsinger/train_svt.py --config configs/comelsinger/svt_train.yaml --smoke-test
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from models.tts.comelsinger.dataset import (
    CoMelSingerDataset,
    comelsinger_collate_fn,
    prepare_batch_for_svt,
)
from models.tts.comelsinger.losses import compute_svt_loss
from models.tts.comelsinger.svt_module import SVTModule
from models.tts.comelsinger.train_utils import (
    check_loss_finite,
    get_device as _get_device,
    load_config as _load_config,
    set_seed as _set_seed,
)

# TensorBoard is optional
try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TENSORBOARD = True
except ImportError:
    _HAS_TENSORBOARD = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# M3-03: Utilities
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="SVT Training Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g. configs/comelsinger/svt_train.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialize everything but skip actual training loop",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short smoke test (max_steps=100, batch_size=4)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Delegates to ``train_utils.load_config`` (shared implementation).
    """
    return _load_config(config_path)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all frameworks.

    Delegates to ``train_utils.set_seed`` (shared implementation).
    """
    _set_seed(seed)


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).

    Delegates to ``train_utils.get_device`` (shared implementation).
    """
    return _get_device()


# ---------------------------------------------------------------------------
# M3-03: Initialization helpers
# ---------------------------------------------------------------------------


def build_model(config: dict) -> SVTModule:
    """Build SVTModule from config dict.

    Args:
        config: Full config dict (uses config["model"] section).

    Returns:
        Initialized SVTModule.
    """
    model_cfg = config["model"]
    return SVTModule(
        num_codebooks=model_cfg.get("num_codebooks", 12),
        codebook_size=model_cfg.get("codebook_size", 1024),
        codebook_embed_dim=model_cfg.get("codebook_embed_dim", 64),
        hidden_size=model_cfg.get("hidden_size", 512),
        num_layers=model_cfg.get("num_layers", 4),
        num_heads=model_cfg.get("num_heads", 8),
        pitch_vocab_size=model_cfg.get("pitch_vocab_size", 129),
        dropout=model_cfg.get("dropout", 0.1),
    )


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build optimizer from config dict.

    Args:
        model: The model whose parameters to optimize.
        config: Full config dict (uses config["optimizer"] section).

    Returns:
        Configured optimizer instance.
    """
    opt_cfg = config["optimizer"]
    opt_type = opt_cfg.get("type", "AdamW")

    if opt_type == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.get("lr", 1e-5),
            weight_decay=opt_cfg.get("weight_decay", 0.01),
        )
    elif opt_type == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg.get("lr", 1e-5),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build learning rate scheduler from config dict.

    Args:
        optimizer: The optimizer to schedule.
        config: Full config dict (uses config["scheduler"] section).

    Returns:
        Configured scheduler or None if not specified.
    """
    sched_cfg = config.get("scheduler")
    if sched_cfg is None:
        return None

    sched_type = sched_cfg.get("type", "cosine")

    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get("T_max", config["training"]["max_steps"]),
            eta_min=sched_cfg.get("eta_min", 0),
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")


def build_dataloaders(
    config: dict, is_smoke_test: bool = False
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader | None]:
    """Build train (and optional val) DataLoaders from config.

    Args:
        config: Full config dict (uses config["data"] section).
        is_smoke_test: If True, reduce batch_size to 4.

    Returns:
        Tuple of (train_loader, val_loader). val_loader may be None if no
        validation split is found.
    """
    data_cfg = config["data"]
    data_dir = Path(data_cfg["data_dir"])
    batch_size = 4 if is_smoke_test else config["training"]["batch_size"]
    num_workers = data_cfg.get("num_workers", 4)

    # Try to find train/val splits
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if train_dir.exists():
        train_dataset = CoMelSingerDataset(str(train_dir))
    else:
        train_dataset = CoMelSingerDataset(str(data_dir))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=comelsinger_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = None
    if val_dir.exists():
        val_dataset = CoMelSingerDataset(str(val_dir))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=comelsinger_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# M3-06: Validation / Evaluation
# ---------------------------------------------------------------------------


def evaluate_svt(
    model: SVTModule,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate SVT model on validation set.

    Computes frame-level accuracy and F1 score by comparing
    argmax(logits) with ground-truth pitch tokens, excluding padded frames.

    Args:
        model: SVTModule (will be set to eval mode internally).
        val_loader: DataLoader yielding collated batches.
        device: Device to run evaluation on.

    Returns:
        dict with keys: "accuracy", "f1", "val_loss"
    """
    model.eval()
    total_correct = 0
    total_frames = 0
    total_loss = 0.0
    num_batches = 0

    # Per-class TP/FP/FN for macro F1
    num_classes = model.pitch_vocab_size
    tp = torch.zeros(num_classes, dtype=torch.long)
    fp = torch.zeros(num_classes, dtype=torch.long)
    fn = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for batch in val_loader:
            batch = prepare_batch_for_svt(batch)
            acoustic_tokens = batch["acoustic_tokens"].to(device)
            pitch_tokens = batch["pitch_tokens"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            note_durations = batch["note_durations"]
            note_pitches = batch["note_pitches"]

            out = model(acoustic_tokens, attention_mask=attention_mask)
            logits = out["logits"]

            # Compute validation loss
            frame_alignment = [d.tolist() for d in note_durations]
            pitch_note_labels = [p.tolist() for p in note_pitches]
            loss, _ = compute_svt_loss(
                pitch_logits=logits,
                target_pitch_tokens=pitch_tokens,
                frame_alignment=frame_alignment,
                pitch_note_labels=pitch_note_labels,
                attention_mask=attention_mask,
            )
            total_loss += loss.item()
            num_batches += 1

            # Frame-level accuracy
            preds = logits.argmax(dim=-1)  # (B, L)
            mask = attention_mask.bool()
            correct = ((preds == pitch_tokens) & mask).sum().item()
            frames = mask.sum().item()
            total_correct += correct
            total_frames += frames

            # Per-class statistics (on CPU for accumulation)
            preds_cpu = preds.cpu()
            targets_cpu = pitch_tokens.cpu()
            mask_cpu = mask.cpu()

            for c in range(num_classes):
                pred_c = (preds_cpu == c) & mask_cpu
                target_c = (targets_cpu == c) & mask_cpu
                tp[c] += (pred_c & target_c).sum().item()
                fp[c] += (pred_c & ~target_c).sum().item()
                fn[c] += (~pred_c & target_c).sum().item()

    accuracy = total_correct / max(total_frames, 1)
    avg_loss = total_loss / max(num_batches, 1)

    # Macro F1: average F1 over classes that appear in ground truth
    precision = tp.float() / (tp + fp).float().clamp(min=1)
    recall = tp.float() / (tp + fn).float().clamp(min=1)
    f1_per_class = 2 * precision * recall / (precision + recall).clamp(min=1e-8)

    # Only average over classes that actually appear in the data
    active_classes = (tp + fn) > 0
    if active_classes.any():
        macro_f1 = f1_per_class[active_classes].mean().item()
    else:
        macro_f1 = 0.0

    model.train()
    return {"accuracy": accuracy, "f1": macro_f1, "val_loss": avg_loss}


# ---------------------------------------------------------------------------
# M3-05: Checkpoint save/load
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: SVTModule,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    step: int,
    best_f1: float,
    output_dir: str | Path,
    filename: str = "checkpoint.pt",
) -> Path:
    """Save training checkpoint.

    Args:
        model: The SVTModule to save.
        optimizer: Optimizer state.
        scheduler: Scheduler state (may be None).
        step: Current training step.
        best_f1: Best validation F1 so far.
        output_dir: Directory to save the checkpoint.
        filename: Checkpoint filename.

    Returns:
        Path to the saved checkpoint file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    state = {
        "step": step,
        "best_f1": best_f1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, path)
    logger.info("Saved checkpoint to %s (step=%d, best_f1=%.4f)", path, step, best_f1)
    return path


def load_checkpoint(
    path: str | Path,
    model: SVTModule,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    device: torch.device | None = None,
) -> dict:
    """Load training checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load state into.
        optimizer: Optimizer to load state into (optional).
        scheduler: Scheduler to load state into (optional).
        device: Device to map tensors to.

    Returns:
        dict with "step" and "best_f1" from checkpoint.
    """
    map_location = device if device is not None else "cpu"
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    logger.info(
        "Loaded checkpoint from %s (step=%d, best_f1=%.4f)",
        path,
        state["step"],
        state["best_f1"],
    )
    return {"step": state["step"], "best_f1": state["best_f1"]}


# ---------------------------------------------------------------------------
# M3-04: Training loop
# ---------------------------------------------------------------------------


def train_step(
    model: SVTModule,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    config: dict,
    device: torch.device,
) -> dict[str, float]:
    """Execute a single training step.

    Args:
        model: SVTModule in train mode.
        batch: Collated batch from DataLoader.
        optimizer: Optimizer instance.
        config: Full config dict (uses config["loss"] and config["training"]).
        device: Device to run on.

    Returns:
        dict with loss values: "loss_total", "l_ce", "l_seg", "l_dur", "lr"
    """
    model.train()
    batch = prepare_batch_for_svt(batch)

    acoustic_tokens = batch["acoustic_tokens"].to(device)
    pitch_tokens = batch["pitch_tokens"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    note_durations = batch["note_durations"]
    note_pitches = batch["note_pitches"]

    # Forward
    out = model(acoustic_tokens, attention_mask=attention_mask)
    logits = out["logits"]  # (B, L, 129)

    # Compute SVT loss
    loss_cfg = config["loss"]
    frame_alignment = [d.tolist() for d in note_durations]
    pitch_note_labels = [p.tolist() for p in note_pitches]

    total_loss, components = compute_svt_loss(
        pitch_logits=logits,
        target_pitch_tokens=pitch_tokens,
        frame_alignment=frame_alignment,
        pitch_note_labels=pitch_note_labels,
        attention_mask=attention_mask,
        lambda_seg=loss_cfg.get("lambda_seg", 3.0),
        lambda_dur=loss_cfg.get("lambda_dur", 5.0),
        delta=loss_cfg.get("delta", 0.5),
    )

    # Backward
    optimizer.zero_grad()
    total_loss.backward()

    # P4: NaN/Inf detection — skip step if loss is non-finite
    if not check_loss_finite(total_loss, step=-1):
        optimizer.zero_grad()
        current_lr = optimizer.param_groups[0]["lr"]
        return {
            "loss_total": float("nan"),
            "l_ce": float("nan"),
            "l_seg": float("nan"),
            "l_dur": float("nan"),
            "lr": current_lr,
        }

    # Gradient clipping
    grad_clip = config["training"].get("gradient_clip", 1.0)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    current_lr = optimizer.param_groups[0]["lr"]
    return {
        "loss_total": total_loss.item(),
        "l_ce": components["l_ce"].item(),
        "l_seg": components["l_seg"].item(),
        "l_dur": components["l_dur"].item(),
        "lr": current_lr,
    }


def infinite_dataloader(dataloader: torch.utils.data.DataLoader):
    """Yield batches infinitely by cycling through the DataLoader."""
    while True:
        for batch in dataloader:
            yield batch


# ---------------------------------------------------------------------------
# M3-04 + M3-05 + M3-06 + M3-07: Main training function
# ---------------------------------------------------------------------------


def train(config: dict, args: argparse.Namespace) -> None:
    """Main training function.

    Implements the full SVT training pipeline:
    - M3-03: Model/optimizer/scheduler initialization
    - M3-04: Training loop with step-based iteration
    - M3-05: Logging (stdout + optional TensorBoard) and checkpointing
    - M3-06: Periodic validation with accuracy/F1 evaluation
    - M3-07: Smoke test mode (max_steps=100, batch_size=4)

    Args:
        config: Loaded YAML config dict.
        args: Parsed command-line arguments.
    """
    is_smoke_test = getattr(args, "smoke_test", False)

    # M3-07: Override config for smoke test
    if is_smoke_test:
        config["training"]["max_steps"] = 100
        config["training"]["batch_size"] = 4
        config["training"]["log_interval"] = 10
        config["training"]["save_interval"] = 50
        config["training"]["val_interval"] = 50
        logger.info("Smoke test mode: max_steps=100, batch_size=4")

    seed = config["training"].get("seed", 42)
    set_seed(seed)

    device = get_device()
    logger.info("Using device: %s", device)

    # Build model
    model = build_model(config)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("SVTModule initialized: %s parameters", f"{n_params:,}")

    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # Dry-run: exit after initialization
    if args.dry_run:
        logger.info("Dry-run complete. Model, optimizer, and scheduler initialized.")
        return

    # Build data loaders
    train_loader, val_loader = build_dataloaders(config, is_smoke_test=is_smoke_test)
    logger.info(
        "DataLoaders built: train=%d batches, val=%s",
        len(train_loader),
        f"{len(val_loader)} batches" if val_loader else "None",
    )

    # Optional TensorBoard writer
    tb_writer = None
    if _HAS_TENSORBOARD:
        tb_log_dir = Path(config["checkpoint"]["output_dir"]) / "tb_logs"
        tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
        logger.info("TensorBoard logging to %s", tb_log_dir)

    # Training state
    max_steps = config["training"]["max_steps"]
    log_interval = config["training"]["log_interval"]
    save_interval = config["training"]["save_interval"]
    val_interval = config["training"]["val_interval"]
    output_dir = config["checkpoint"]["output_dir"]
    best_f1 = 0.0
    start_step = 0

    # Resume from checkpoint
    if args.resume:
        ckpt_info = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        start_step = ckpt_info["step"]
        best_f1 = ckpt_info["best_f1"]
        logger.info("Resuming from step %d (best_f1=%.4f)", start_step, best_f1)

    # M3-04: Main training loop (step-based)
    model.train()
    batch_iter = infinite_dataloader(train_loader)
    t_start = time.time()

    for step in range(start_step, max_steps):
        batch = next(batch_iter)

        metrics = train_step(model, batch, optimizer, config, device)

        if scheduler is not None:
            scheduler.step()

        # M3-05: Logging
        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t_start
            steps_per_sec = (step + 1 - start_step) / max(elapsed, 1e-6)
            logger.info(
                "step=%d/%d  loss=%.4f  l_ce=%.4f  l_seg=%.4f  l_dur=%.4f  "
                "lr=%.2e  steps/s=%.2f",
                step + 1,
                max_steps,
                metrics["loss_total"],
                metrics["l_ce"],
                metrics["l_seg"],
                metrics["l_dur"],
                metrics["lr"],
                steps_per_sec,
            )

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss_total", metrics["loss_total"], step + 1)
                tb_writer.add_scalar("train/l_ce", metrics["l_ce"], step + 1)
                tb_writer.add_scalar("train/l_seg", metrics["l_seg"], step + 1)
                tb_writer.add_scalar("train/l_dur", metrics["l_dur"], step + 1)
                tb_writer.add_scalar("train/lr", metrics["lr"], step + 1)

        # M3-05: Save checkpoint at intervals
        if (step + 1) % save_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                step + 1,
                best_f1,
                output_dir,
                filename=f"checkpoint_step{step + 1}.pt",
            )

        # M3-06: Validation at intervals
        if val_loader is not None and (step + 1) % val_interval == 0:
            val_metrics = evaluate_svt(model, val_loader, device)
            logger.info(
                "Validation step=%d: accuracy=%.4f  f1=%.4f  val_loss=%.4f",
                step + 1,
                val_metrics["accuracy"],
                val_metrics["f1"],
                val_metrics["val_loss"],
            )

            if tb_writer is not None:
                tb_writer.add_scalar("val/accuracy", val_metrics["accuracy"], step + 1)
                tb_writer.add_scalar("val/f1", val_metrics["f1"], step + 1)
                tb_writer.add_scalar("val/loss", val_metrics["val_loss"], step + 1)

            # Save best model by F1
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step + 1,
                    best_f1,
                    output_dir,
                    filename="svt_best.pt",
                )
                logger.info("New best F1: %.4f (saved as svt_best.pt)", best_f1)

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, max_steps, best_f1, output_dir, filename="svt_final.pt"
    )
    logger.info("Training complete. Final step: %d, best F1: %.4f", max_steps, best_f1)

    if tb_writer is not None:
        tb_writer.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for SVT training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    config = load_config(args.config)
    train(config, args)


if __name__ == "__main__":
    main()
