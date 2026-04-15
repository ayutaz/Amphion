"""S2A (Semantic-to-Acoustic) Training Script -- Algorithm 1.

Trains CoMelSinger_S2A with:
- Contrastive Learning (SCL + FCL) for prosody leakage prevention
- SVT pitch supervision (frozen)
- Mask prediction loss (MaskGCT base)
- LoRA fine-tuning

Paper Section III-A/B/C, IV-B.

Usage:
    uv run python models/tts/comelsinger/train_s2a.py --config configs/comelsinger/s2a_train.yaml
    uv run python models/tts/comelsinger/train_s2a.py --config configs/comelsinger/s2a_train.yaml --dry-run
"""
from __future__ import annotations

import argparse
import logging
import math
import os
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# M3-22: Expected lambda values for cross-check assertions
# These MUST match configs/comelsinger/s2a_train.yaml (Section IV-B confirmed)
# ---------------------------------------------------------------------------
EXPECTED_LAMBDAS = {
    "lambda_cl": 0.5,
    "lambda_scl": 1.0,
    "lambda_fcl": 0.1,
    "lambda_svt": 0.5,
    "lambda_mask": 0.3,
    "lambda_seg": 3.0,
    "lambda_dur": 5.0,
    "tau": 0.07,
}


# ===== M3-10: Pitch perturbation ===========================================

def pitch_perturbation(
    pitch_tokens: torch.Tensor,
    zero_prob: float = 0.5,
    max_shift: int = 6,
    pitch_vocab_size: int = 129,
) -> torch.Tensor:
    """Apply random pitch perturbation per sample (not batch-wide).

    Each sample independently: with prob *zero_prob*, set all to 0;
    else random shift in [-max_shift, +max_shift].

    Args:
        pitch_tokens: (B, T) pitch token tensor.
        zero_prob: probability of zeroing out each sample.
        max_shift: maximum semitone shift magnitude.
        pitch_vocab_size: upper clamp value (exclusive).

    Returns:
        perturbed: (B, T), same dtype/device.  Original tensor is NOT modified.
    """
    B = pitch_tokens.shape[0]
    perturbed = pitch_tokens.clone()
    for b in range(B):
        if torch.rand(1).item() < zero_prob:
            perturbed[b] = 0
        else:
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            perturbed[b] = (pitch_tokens[b] + shift).clamp(0, pitch_vocab_size - 1)
    return perturbed


# ===== M3-11: Prompt generation =============================================

def prompt_gen(
    batch: Dict[str, torch.Tensor],
    max_len: int = 150,
) -> Dict[str, torch.Tensor]:
    """Generate acoustic prompts from same-speaker samples.

    For each sample, selects a *different* sample from the same speaker as
    the timbre prompt.  Falls back to a random other sample if no same-speaker
    candidate exists.

    Args:
        batch: collated batch dict containing at least ``speaker_id`` (B,)
               and tensor values with dim >= 1 whose first axis is batch.
        max_len: maximum prompt frame length (truncates dim-1 tensors).

    Returns:
        prompt_batch: dict with the same keys, values indexed by selected
                      prompt indices and truncated to *max_len* on axis 1.
    """
    speaker_ids = batch["speaker_id"]  # (B,)
    B = speaker_ids.size(0)
    prompt_indices = torch.zeros(B, dtype=torch.long, device=speaker_ids.device)

    for i in range(B):
        same_spk = (speaker_ids == speaker_ids[i]).nonzero(as_tuple=True)[0]
        candidates = same_spk[same_spk != i]
        if len(candidates) > 0:
            prompt_indices[i] = candidates[torch.randint(len(candidates), (1,))]
        else:
            others = [j for j in range(B) if j != i]
            prompt_indices[i] = others[torch.randint(len(others), (1,)).item()]

    prompt_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            indexed = v[prompt_indices]
            if indexed.dim() == 2:
                # (B, T) -> truncate time dim
                prompt_batch[k] = indexed[:, :max_len]
            elif indexed.dim() == 3:
                # (B, C, T) like acoustic_tokens -> truncate last dim
                prompt_batch[k] = indexed[:, :, :max_len]
            else:
                prompt_batch[k] = indexed
        elif isinstance(v, list):
            prompt_batch[k] = [v[idx.item()] for idx in prompt_indices]
        else:
            prompt_batch[k] = v
    return prompt_batch


# ===== M3-09: Batch split ==================================================

def split_batch(
    batch: Dict[str, Any],
    k_s: int,
) -> tuple:
    """Split batch into SCL subset (first k_s) and FCL subset (rest).

    Args:
        batch: collated batch dict with tensor values (B, ...).
        k_s: number of samples for SCL (first k_s samples).

    Returns:
        (s_a_s, s_a_f): two dicts with the same keys.
    """
    s_a_s: Dict[str, Any] = {}
    s_a_f: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            s_a_s[k] = v[:k_s]
            s_a_f[k] = v[k_s:]
        elif isinstance(v, list):
            s_a_s[k] = v[:k_s]
            s_a_f[k] = v[k_s:]
        else:
            s_a_s[k] = v
            s_a_f[k] = v
    return s_a_s, s_a_f


# ===== M3-14: Soft label matrix builder =====================================

def build_soft_label_matrix(pitch_tokens: torch.Tensor) -> torch.Tensor:
    """Build soft label matrices for FCL from pitch tokens.

    Args:
        pitch_tokens: (B, T) pitch token tensor.

    Returns:
        Y: (B, T, T) soft label matrix.
           Y[b, i, j] = 1.0 if frames i and j share the same nonzero pitch.
    """
    # (B, T)
    voiced = (pitch_tokens != 0)  # (B, T) bool
    voiced_2d = voiced.unsqueeze(2) & voiced.unsqueeze(1)  # (B, T, T)
    same_pitch = (pitch_tokens.unsqueeze(2) == pitch_tokens.unsqueeze(1))  # (B, T, T)
    Y = (voiced_2d & same_pitch).float()
    return Y


# ===== Inverse square-root scheduler ========================================

def inverse_sqrt_schedule(step: int, warmup_steps: int = 1000) -> float:
    """Inverse square-root LR multiplier with linear warmup.

    During warmup (step < warmup_steps):   lr_mult = (step+1) / warmup_steps
    After warmup:                          lr_mult = sqrt(warmup_steps) / sqrt(step+1)
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return math.sqrt(warmup_steps) / math.sqrt(step + 1)


# ===== M3-08: Model & optimizer construction ================================

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_s2a_model(cfg: dict, device: torch.device):
    """Build CoMelSinger_S2A and optionally load pretrained MaskGCT-S2A weights.

    Returns the raw (non-LoRA) model.
    """
    from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A

    model_cfg = cfg.get("model", {})
    model = CoMelSinger_S2A(
        pitch_vocab_size=model_cfg.get("pitch_vocab_size", 129),
        temperature=model_cfg.get("temperature", 0.07),
        lambda_cl=cfg.get("loss", {}).get("lambda_cl", 0.5),
        lambda_scl=cfg.get("loss", {}).get("lambda_scl", 1.0),
        lambda_fcl=cfg.get("loss", {}).get("lambda_fcl", 0.1),
        lambda_svt=cfg.get("loss", {}).get("lambda_svt", 0.5),
        lambda_mask=cfg.get("loss", {}).get("lambda_mask", 0.3),
    )

    # Optional pretrained weight loading (strict=False for new pitch_emb keys)
    pretrained_path = cfg.get("pretrained", {}).get("s2a_path")
    if pretrained_path and os.path.exists(pretrained_path):
        logger.info("Loading pretrained S2A weights from %s", pretrained_path)
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing:
            logger.info("Missing keys (expected for new layers): %s", missing)
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected)
    else:
        logger.info("No pretrained S2A path provided or file not found; training from scratch.")

    return model.to(device)


def apply_lora(model, cfg: dict):
    """Apply LoRA adapter to the model and return a PeftModel.

    modules_to_save includes pitch_emb so it stays fully trainable.
    """
    from peft import LoraConfig, get_peft_model

    lora_cfg = cfg.get("lora", {})
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
        modules_to_save=lora_cfg.get("modules_to_save", ["pitch_emb"]),
    )
    peft_model = get_peft_model(model, lora_config)

    total = sum(p.numel() for p in peft_model.parameters())
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    logger.info(
        "LoRA applied: total=%d, trainable=%d (%.2f%%)",
        total, trainable, 100.0 * trainable / max(total, 1),
    )
    return peft_model


def build_svt_model(cfg: dict, device: torch.device):
    """Load SVT module and freeze all parameters (StopGrad)."""
    from models.tts.comelsinger.svt_module import SVTModule

    svt = SVTModule()
    svt_ckpt_path = cfg.get("checkpoint", {}).get("svt_checkpoint")
    if svt_ckpt_path and os.path.exists(svt_ckpt_path):
        logger.info("Loading SVT checkpoint from %s", svt_ckpt_path)
        svt.load_state_dict(
            torch.load(svt_ckpt_path, map_location="cpu", weights_only=True)
        )
    else:
        logger.info("No SVT checkpoint found; using randomly initialized SVT.")

    svt = svt.to(device)
    svt.freeze()  # requires_grad_(False) + eval()
    return svt


def build_optimizer_and_scheduler(model, cfg: dict):
    """Build AdamW optimizer and inverse-sqrt LR scheduler."""
    opt_cfg = cfg.get("optimizer", {})
    sched_cfg = cfg.get("scheduler", {})

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_cfg.get("lr", 1e-5),
        weight_decay=opt_cfg.get("weight_decay", 0.01),
    )

    warmup_steps = sched_cfg.get("warmup_steps", 1000)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: inverse_sqrt_schedule(step, warmup_steps),
    )
    return optimizer, scheduler


# ===== M3-22: Config cross-check assertions =================================

def crosscheck_config(cfg: dict) -> None:
    """Assert that config lambda values match EXPECTED_LAMBDAS.

    Raises AssertionError with a clear message if any mismatch is found.
    """
    loss_cfg = cfg.get("loss", {})
    for key, expected in EXPECTED_LAMBDAS.items():
        actual = loss_cfg.get(key)
        if actual is None:
            continue  # key not in config section; skip optional ones
        assert actual == expected, (
            f"Config cross-check FAILED: loss.{key} = {actual}, expected {expected}"
        )
    logger.info("Config cross-check PASSED: all lambda values match.")


# ===== M3-12/13/14/15/16/17/18: Algorithm 1 training step ==================

def algorithm1_step(
    s2a_model: nn.Module,
    svt_model: nn.Module,
    batch: Dict[str, Any],
    cfg: dict,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Execute one training step of Algorithm 1.

    This function implements the full Algorithm 1 from paper Section IV-B:
    1. Split batch into SCL (k_s) and FCL (K-k_s) subsets
    2. Apply pitch perturbation
    3. Compute S2A forward with original and perturbed pitch (cond_B, cond_Bp)
    4. Compute SCL loss (sequence-level contrastive)
    5. Compute FCL loss (frame-level contrastive)
    6. Compute L_CL = lambda_scl * L_SCL + lambda_fcl * L_FCL
    7. Compute L_mask via S2A compute_loss
    8. Compute L_SVT via frozen SVT
    9. Compute L_total = lambda_cl * L_CL + lambda_svt * L_SVT + lambda_mask * L_mask

    Args:
        s2a_model: CoMelSinger_S2A (possibly wrapped in PeftModel / DDP).
        svt_model: frozen SVTModule.
        batch: collated batch dict from DataLoader.
        cfg: full config dict.
        device: computation device.

    Returns:
        dict with keys: l_total, l_scl, l_fcl, l_cl, l_svt, l_mask, l_ce, l_seg, l_dur
    """
    from models.tts.comelsinger.losses import (
        compute_fcl_loss,
        compute_scl_loss,
        compute_svt_loss,
        compute_total_loss,
    )

    loss_cfg = cfg.get("loss", {})
    training_cfg = cfg.get("training", {})
    k_s = training_cfg.get("k_s", 8)
    tau = loss_cfg.get("tau", 0.07)
    lambda_cl = loss_cfg.get("lambda_cl", 0.5)
    lambda_scl = loss_cfg.get("lambda_scl", 1.0)
    lambda_fcl = loss_cfg.get("lambda_fcl", 0.1)
    lambda_svt = loss_cfg.get("lambda_svt", 0.5)
    lambda_mask = loss_cfg.get("lambda_mask", 0.3)
    lambda_seg = loss_cfg.get("lambda_seg", 3.0)
    lambda_dur = loss_cfg.get("lambda_dur", 5.0)
    delta = loss_cfg.get("delta", 0.5)

    # -- Move batch tensors to device --
    def to_dev(v):
        return v.to(device) if isinstance(v, torch.Tensor) else v

    batch = {k: to_dev(v) for k, v in batch.items()}

    # Access the underlying model through potential PeftModel / DDP wrappers
    base_model = s2a_model
    if hasattr(base_model, "module"):
        base_model = base_model.module
    # PeftModel wraps the actual CoMelSinger_S2A as .model or .base_model
    if hasattr(base_model, "get_base_model"):
        _inner = base_model.get_base_model()
    else:
        _inner = base_model

    # Prepare tensors: acoustic_tokens from collate is (B, 12, T) -> need (B, T, 12)
    acoustic_tokens = batch["acoustic_tokens"]
    if acoustic_tokens.dim() == 3 and acoustic_tokens.shape[1] == 12:
        acoustic_tokens = acoustic_tokens.permute(0, 2, 1)  # (B, T, 12)
    semantic_tokens = batch["semantic_tokens"]  # (B, T)
    pitch_tokens = batch["pitch_tokens"]  # (B, T)
    attention_mask = batch["attention_mask"]  # (B, T)

    B, T = semantic_tokens.shape

    # ===== M3-10: Pitch perturbation ========================================
    perturbed_pitch = pitch_perturbation(
        pitch_tokens,
        zero_prob=0.5,
        max_shift=6,
        pitch_vocab_size=129,
    )

    # ===== M3-12: S2A forward (cond_B and cond_Bp) ==========================
    # get_cond builds: cond_emb(semantic) + pitch_emb(pitch)
    cond_B = _inner.get_cond(semantic_tokens, pitch_tokens)  # (B, T, D)
    cond_Bp = _inner.get_cond(semantic_tokens, perturbed_pitch)  # (B, T, D)

    # ===== M3-09: Batch split ===============================================
    # Ensure k_s does not exceed batch size
    actual_k_s = min(k_s, B)

    # ===== M3-13: SCL loss ==================================================
    # Global average pool -> (K_s, D)
    if actual_k_s >= 2:
        g_a = cond_B[:actual_k_s].mean(dim=1)   # (K_s, D)
        g_b = cond_Bp[:actual_k_s].mean(dim=1)  # (K_s, D)
        l_scl = compute_scl_loss(g_a, g_b, tau=tau)
    else:
        l_scl = torch.tensor(0.0, device=device, requires_grad=True)

    # ===== M3-14: FCL loss ==================================================
    fcl_start = actual_k_s
    if fcl_start < B:
        f_a = cond_B[fcl_start:]   # (K-K_s, T, D)
        f_b = cond_Bp[fcl_start:]  # (K-K_s, T, D)
        Y = build_soft_label_matrix(pitch_tokens[fcl_start:])  # (K-K_s, T, T)
        l_fcl = compute_fcl_loss(f_a, f_b, Y, tau=tau)
    else:
        l_fcl = torch.tensor(0.0, device=device, requires_grad=True)

    # ===== M3-15: L_CL =====================================================
    l_cl = lambda_scl * l_scl + lambda_fcl * l_fcl

    # ===== M3-16: L_mask (MaskGCT mask prediction loss) =====================
    # CoMelSinger_S2A.forward -> compute_loss -> loss_t
    # Returns: logits, mask_layer, final_mask, x0, prompt_len, mask_prob
    logits, mask_layer, final_mask, x0_out, prompt_len, mask_prob = s2a_model(
        acoustic_tokens, attention_mask.float(), semantic_tokens, pitch_tokens
    )
    # Compute cross-entropy on masked positions
    target = x0_out[:, :, mask_layer.item()]  # (B, T) target codes for this layer
    # final_mask: (B, T, 1) -> squeeze
    mask_flat = final_mask.squeeze(-1).bool()  # (B, T)
    if mask_flat.any():
        l_mask = F.cross_entropy(
            logits[mask_flat],  # (N, codebook_size)
            target[mask_flat],  # (N,)
        )
    else:
        l_mask = torch.tensor(0.0, device=device, requires_grad=True)

    # ===== M3-17: L_SVT (frozen SVT pitch supervision) ======================
    with torch.no_grad():
        svt_out = svt_model(acoustic_tokens)  # {"logits": (B, T, 129)}

    # Build frame_alignment and pitch_note_labels from batch
    note_durations = batch.get("note_durations", [])
    note_pitches = batch.get("note_pitches", [])

    # If note-level annotations exist, compute full SVT loss
    has_note_info = (
        isinstance(note_durations, list)
        and len(note_durations) == B
        and all(len(nd) > 0 for nd in note_durations)
    )

    if has_note_info:
        frame_alignment = [nd.tolist() if isinstance(nd, torch.Tensor) else nd for nd in note_durations]
        pitch_note_labels = [np_item.tolist() if isinstance(np_item, torch.Tensor) else np_item for np_item in note_pitches]

        l_svt_total, svt_detail = compute_svt_loss(
            pitch_logits=svt_out["logits"].detach(),
            target_pitch_tokens=pitch_tokens,
            frame_alignment=frame_alignment,
            pitch_note_labels=pitch_note_labels,
            attention_mask=attention_mask,
            lambda_seg=lambda_seg,
            lambda_dur=lambda_dur,
            delta=delta,
        )
    else:
        # Fallback: CE-only SVT loss (no note-level alignment available)
        svt_logits = svt_out["logits"].detach()
        targets_svt = pitch_tokens.clone()
        if attention_mask is not None:
            targets_svt[~attention_mask.bool()] = -100
        l_svt_total = F.cross_entropy(
            svt_logits.reshape(-1, svt_logits.size(-1)),
            targets_svt.reshape(-1),
            ignore_index=-100,
        )
        svt_detail = {"l_ce": l_svt_total, "l_seg": torch.tensor(0.0), "l_dur": torch.tensor(0.0)}

    # SVT loss is detached (no grad through SVT), but we keep it as a
    # differentiable scalar so it can be part of the total loss graph
    # (even though its gradient contribution is zero for SVT params).
    l_svt = l_svt_total.detach()

    # ===== M3-18: Total loss ================================================
    l_total, loss_dict = compute_total_loss(
        l_scl=l_scl,
        l_fcl=l_fcl,
        l_svt=l_svt,
        l_mask=l_mask,
        lambda_cl=lambda_cl,
        lambda_scl=lambda_scl,
        lambda_fcl=lambda_fcl,
        lambda_svt=lambda_svt,
        lambda_mask=lambda_mask,
    )

    # Merge SVT sub-losses into output dict
    loss_dict["l_ce"] = svt_detail["l_ce"].detach() if isinstance(svt_detail["l_ce"], torch.Tensor) else svt_detail["l_ce"]
    loss_dict["l_seg"] = svt_detail["l_seg"].detach() if isinstance(svt_detail["l_seg"], torch.Tensor) else svt_detail["l_seg"]
    loss_dict["l_dur"] = svt_detail["l_dur"].detach() if isinstance(svt_detail["l_dur"], torch.Tensor) else svt_detail["l_dur"]

    return l_total, loss_dict


# ===== M3-19: TensorBoard logging ==========================================

def log_metrics(
    writer,
    loss_dict: Dict[str, Any],
    lr: float,
    grad_norm: float,
    global_step: int,
    is_main_process: bool = True,
) -> None:
    """Log losses and training stats to TensorBoard (main process only).

    Args:
        writer: SummaryWriter instance (or None).
        loss_dict: dict from algorithm1_step / compute_total_loss.
        lr: current learning rate.
        grad_norm: gradient norm after clipping.
        global_step: global training step.
        is_main_process: whether this is the main (rank-0) process.
    """
    if not is_main_process or writer is None:
        return

    log_keys = ["l_scl", "l_fcl", "l_cl", "l_svt", "l_mask", "l_total",
                "l_ce", "l_seg", "l_dur"]
    for key in log_keys:
        val = loss_dict.get(key)
        if val is not None:
            scalar = val.item() if isinstance(val, torch.Tensor) else float(val)
            writer.add_scalar(f"train/{key}", scalar, global_step)

    writer.add_scalar("train/lr", lr, global_step)
    writer.add_scalar("train/grad_norm", grad_norm, global_step)


# ===== M3-20: Checkpoint saving ============================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    loss_dict: Dict[str, Any],
    output_dir: str,
    is_main_process: bool = True,
) -> None:
    """Save LoRA adapter weights and training state.

    Saves every save_interval epochs.  Uses PEFT save_pretrained if available,
    otherwise falls back to torch.save.

    Args:
        model: PeftModel or raw model.
        optimizer: optimizer.
        scheduler: LR scheduler.
        epoch: current epoch (0-based).
        global_step: global step count.
        loss_dict: latest loss dict for metadata.
        output_dir: base checkpoint directory.
        is_main_process: only save on rank-0.
    """
    if not is_main_process:
        return

    epoch_dir = os.path.join(output_dir, f"epoch_{epoch:04d}")
    os.makedirs(epoch_dir, exist_ok=True)

    # Save LoRA adapter if model is a PeftModel
    unwrapped = model.module if hasattr(model, "module") else model
    if hasattr(unwrapped, "save_pretrained"):
        unwrapped.save_pretrained(epoch_dir)
        logger.info("Saved LoRA adapter to %s", epoch_dir)
    else:
        # Fallback: save full state dict
        torch.save(unwrapped.state_dict(), os.path.join(epoch_dir, "model.pt"))
        logger.info("Saved model state_dict to %s", epoch_dir)

    # Save optimizer + scheduler state for resumption
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        os.path.join(epoch_dir, "training_state.pt"),
    )


# ===== Main training loop ==================================================

def train(cfg: dict, dry_run: bool = False) -> None:
    """Main S2A training loop implementing Algorithm 1.

    Args:
        cfg: full configuration dict loaded from YAML.
        dry_run: if True, only build model and run 1 step, then exit.
    """
    # ---- M3-22: Cross-check config lambdas ----
    crosscheck_config(cfg)

    training_cfg = cfg.get("training", {})
    seed = training_cfg.get("seed", 42)
    torch.manual_seed(seed)

    # ---- Determine device & precision ----
    precision = training_cfg.get("precision", "bf16")
    use_accelerate = True
    accelerator = None

    try:
        from accelerate import Accelerator
        from accelerate.utils import DDPKwargsHandler

        ddp_kwargs = DDPKwargsHandler(find_unused_parameters=True)
        mixed_prec = precision if precision in ("bf16", "fp16") else "no"
        accelerator = Accelerator(
            mixed_precision=mixed_prec,
            kwargs_handlers=[ddp_kwargs],
        )
        device = accelerator.device
        is_main = accelerator.is_main_process
        logger.info("Using Accelerate: device=%s, precision=%s", device, mixed_prec)
    except ImportError:
        logger.warning("accelerate not installed; falling back to CPU/single-GPU mode.")
        use_accelerate = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
        logger.info("Using device: %s", device)

    # ---- M3-08: Build models ----
    s2a_model = build_s2a_model(cfg, device)

    try:
        s2a_model = apply_lora(s2a_model, cfg)
    except ImportError:
        logger.warning("peft not installed; training without LoRA.")

    svt_model = build_svt_model(cfg, device)

    # ---- Build optimizer & scheduler ----
    optimizer, scheduler = build_optimizer_and_scheduler(s2a_model, cfg)

    # ---- Build DataLoader ----
    from models.tts.comelsinger.dataset import (
        CoMelSingerDataset,
        BalancedSpeakerSampler,
        comelsinger_collate_fn,
    )

    data_cfg = cfg.get("data", {})
    batch_size = training_cfg.get("batch_size", 32)
    k_s = training_cfg.get("k_s", 8)

    try:
        dataset = CoMelSingerDataset(
            data_dir=data_cfg.get("data_dir", "data/preprocessed"),
        )
        sampler = BalancedSpeakerSampler(
            speaker_ids=dataset.speaker_ids,
            batch_size=batch_size,
            k_s=k_s,
            seed=seed,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=comelsinger_collate_fn,
            num_workers=data_cfg.get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
        )
    except (ValueError, FileNotFoundError) as e:
        if dry_run:
            logger.warning("Dataset not available for dry-run: %s. Using dummy data.", e)
            dataloader = None
        else:
            raise

    # ---- Accelerate prepare ----
    if use_accelerate and accelerator is not None:
        if dataloader is not None:
            s2a_model, optimizer, dataloader, scheduler = accelerator.prepare(
                s2a_model, optimizer, dataloader, scheduler,
            )
        else:
            s2a_model, optimizer, scheduler = accelerator.prepare(
                s2a_model, optimizer, scheduler,
            )

    # ---- TensorBoard writer ----
    writer = None
    if is_main:
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = os.path.join(
                cfg.get("checkpoint", {}).get("output_dir", "checkpoints/s2a"),
                "logs",
            )
            writer = SummaryWriter(log_dir=log_dir)
            logger.info("TensorBoard logs: %s", log_dir)
        except ImportError:
            logger.warning("tensorboard not installed; logging disabled.")

    # ---- Dry-run mode ----
    if dry_run:
        logger.info("=== DRY RUN: running 1 step with dummy data ===")
        B_dummy = min(batch_size, 4)
        T_dummy = 50
        dummy_batch = {
            "acoustic_tokens": torch.randint(0, 1024, (B_dummy, 12, T_dummy), device=device),
            "semantic_tokens": torch.randint(0, 1024, (B_dummy, T_dummy), device=device),
            "pitch_tokens": torch.randint(0, 129, (B_dummy, T_dummy), device=device),
            "attention_mask": torch.ones(B_dummy, T_dummy, dtype=torch.long, device=device),
            "speaker_id": torch.randint(0, 3, (B_dummy,), device=device),
            "note_durations": [torch.tensor([10, 10, 10, 10, 10]) for _ in range(B_dummy)],
            "note_pitches": [torch.tensor([60, 62, 64, 65, 67]) for _ in range(B_dummy)],
        }

        s2a_model.train()
        l_total, loss_dict = algorithm1_step(
            s2a_model, svt_model, dummy_batch, cfg, device,
        )

        optimizer.zero_grad()
        if use_accelerate and accelerator is not None:
            accelerator.backward(l_total)
        else:
            l_total.backward()
        optimizer.step()
        scheduler.step()

        logger.info("Dry-run losses: %s",
                     {k: (v.item() if isinstance(v, torch.Tensor) else v)
                      for k, v in loss_dict.items()})
        logger.info("=== DRY RUN COMPLETE ===")

        if writer is not None:
            writer.close()
        return

    # ---- Training loop ----
    max_epochs = training_cfg.get("max_epochs", 100)
    gradient_clip = training_cfg.get("gradient_clip", 1.0)
    log_interval = training_cfg.get("log_interval", 100)
    save_interval = training_cfg.get("save_interval", 10)
    output_dir = cfg.get("checkpoint", {}).get("output_dir", "checkpoints/s2a")

    if dataloader is None:
        raise RuntimeError("No dataloader available. Check data_dir in config.")

    global_step = 0
    s2a_model.train()

    for epoch in range(max_epochs):
        epoch_losses: Dict[str, float] = {}
        num_batches = 0

        for batch in dataloader:
            # ---- Algorithm 1 step ----
            l_total, loss_dict = algorithm1_step(
                s2a_model, svt_model, batch, cfg, device,
            )

            # ---- Backward + update ----
            optimizer.zero_grad()
            if use_accelerate and accelerator is not None:
                accelerator.backward(l_total)
            else:
                l_total.backward()

            # Gradient clipping
            if use_accelerate and accelerator is not None:
                accelerator.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, s2a_model.parameters()),
                    gradient_clip,
                )
                grad_norm = 0.0  # accelerate does not return grad_norm easily
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, s2a_model.parameters()),
                    gradient_clip,
                ).item()

            optimizer.step()
            scheduler.step()
            global_step += 1

            # Accumulate epoch losses
            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                epoch_losses[k] = epoch_losses.get(k, 0.0) + val
            num_batches += 1

            # ---- M3-19: Step-level logging ----
            if global_step % log_interval == 0:
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else 0.0
                log_metrics(
                    writer, loss_dict, current_lr, grad_norm, global_step,
                    is_main_process=is_main,
                )
                if is_main:
                    loss_str = ", ".join(
                        f"{k}={v.item() if isinstance(v, torch.Tensor) else v:.4f}"
                        for k, v in loss_dict.items()
                    )
                    logger.info(
                        "Epoch %d step %d: %s, lr=%.2e",
                        epoch, global_step, loss_str, current_lr,
                    )

        # ---- Epoch-level logging ----
        if is_main and num_batches > 0:
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else 0.0
            log_metrics(
                writer, avg_losses, current_lr, 0.0, global_step,
                is_main_process=is_main,
            )
            avg_str = ", ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
            logger.info("Epoch %d complete: %s", epoch, avg_str)

        # ---- M3-20: Checkpoint saving ----
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                model=s2a_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                global_step=global_step,
                loss_dict=loss_dict if num_batches > 0 else {},
                output_dir=output_dir,
                is_main_process=is_main,
            )

    # Final save
    if is_main:
        save_checkpoint(
            model=s2a_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=max_epochs,
            global_step=global_step,
            loss_dict={},
            output_dir=output_dir,
            is_main_process=is_main,
        )

    if writer is not None:
        writer.close()

    logger.info("Training complete. Total steps: %d", global_step)


# ===== Entry point ==========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoMelSinger S2A Training (Algorithm 1)")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/comelsinger/s2a_train.yaml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build model and run 1 step with dummy data, then exit.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config)
    train(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
