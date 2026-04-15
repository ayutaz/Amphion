"""CoMelSinger loss functions.

Loss composition (paper Section IV-B confirmed values):
    L_CL   = 1.0 * L_SCL + 0.1 * L_FCL
    L_SVT  = L_CE + 3 * L_seg + 5 * L_dur
    L_total = 0.5 * L_CL + 0.5 * L_SVT + 0.3 * L_mask
"""
import torch
import torch.nn.functional as F


def compute_scl_loss(
    g_a: torch.Tensor, g_b: torch.Tensor, tau: float = 0.07
) -> torch.Tensor:
    """NT-Xent symmetric contrastive loss (Sequence-level).

    Args:
        g_a, g_b: (K_s, D) sequence embeddings. K_s=8 typically.
        tau: temperature (default 0.07)
    Returns:
        scalar loss

    Implementation:
        1. L2 normalize g_a, g_b
        2. sim = matmul(g_a, g_b.T) / tau -> (K_s, K_s)
        3. labels = arange(K_s)
        4. loss = (cross_entropy(sim, labels) + cross_entropy(sim.T, labels)) / 2

    NOTE: Cast to FP32 for numerical stability with small tau and FP16 inputs.
    """
    # Cast to FP32 for numerical stability
    g_a = g_a.float()
    g_b = g_b.float()

    g_a = F.normalize(g_a, dim=-1)
    g_b = F.normalize(g_b, dim=-1)

    sim = torch.matmul(g_a, g_b.T) / tau  # (K_s, K_s)
    labels = torch.arange(g_a.size(0), device=g_a.device)

    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0
    return loss


def compute_fcl_loss(
    f_a: torch.Tensor,
    f_b: torch.Tensor,
    Y: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """Soft-label frame-level contrastive loss.

    Args:
        f_a, f_b: (B, L, D) frame embeddings
        Y: (B, L, L) soft label matrix. Y[i,j] in {+1, 0}
           +1 = same pitch & both voiced, 0 = otherwise
        tau: temperature
    Returns:
        scalar loss. Returns 0.0 if Y has no valid entries.

    Implementation:
        1. L2 normalize f_a, f_b
        2. S = bmm(f_a, f_b.T) / tau -> (B, L, L)
        3. valid = (Y != 0).float()
        4. n_valid = valid.sum().clamp(min=1.0)
        5. loss = -(valid * Y * S).sum() / n_valid
    """
    B, L, D = f_a.shape

    # Cast to FP32 for numerical stability with small tau and FP16 inputs
    f_a = f_a.float()
    f_b = f_b.float()
    Y = Y.float()

    f_a = F.normalize(f_a, dim=-1)  # (B, L, D)
    f_b = F.normalize(f_b, dim=-1)  # (B, L, D)

    sim = torch.bmm(f_a, f_b.transpose(1, 2)) / tau  # (B, L, L)
    log_softmax = F.log_softmax(sim, dim=-1)  # (B, L, L)

    # Rows with no positive labels are excluded from the loss
    row_sum = Y.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, L, 1)
    soft_label = Y / row_sum  # (B, L, L)

    # valid_mask: 1 for frames that have at least one positive label
    valid_mask = (Y.sum(dim=-1) > 0).float()  # (B, L)

    per_frame_loss = -(soft_label * log_softmax).sum(dim=-1) * valid_mask  # (B, L)
    valid_count = valid_mask.sum().clamp(min=1.0)
    return per_frame_loss.sum() / valid_count


def compute_svt_loss(
    pitch_probs: torch.Tensor,
    target_pitch_tokens: torch.Tensor,
    frame_alignment: list,
    pitch_note_labels: list,
    attention_mask: torch.Tensor = None,
    lambda_seg: float = 3.0,
    lambda_dur: float = 5.0,
    delta: float = 0.5,
) -> tuple:
    """SVT pitch supervision loss (3 components).

    Args:
        pitch_probs: (B, L, C) softmax probabilities from SVT
        target_pitch_tokens: (B, L) ground truth pitch tokens
        frame_alignment: list of lists, each inner list = frame counts per note [a_d_1, a_d_2, ...]
        pitch_note_labels: list of lists, each inner list = pitch token per note [m_p_1, m_p_2, ...]
        attention_mask: (B, L) 1=valid, 0=padding. If None, all valid.
        lambda_seg, lambda_dur, delta: hyperparameters
    Returns:
        (total_loss, {"l_ce": ..., "l_seg": ..., "l_dur": ...})

    L_CE: CrossEntropy (padding excluded via attention_mask)
    L_seg: paper eq.(7) Euclidean distance based
        sum_t [(1-b_t)*||p_t - p_{t-1}||^2 + b_t * max(0, delta - ||p_t - p_{t-1}||^2)]
        where b_t = 1 if note boundary at t, else 0
    L_dur: paper eq.(8) soft duration
        sum_i (sum_{t=T_i}^{T_i+a_d_i-1} p_t[m_p_i] - a_d_i)^2
    """
    B, L, C = pitch_probs.shape

    # ---- L_CE: CrossEntropy (exclude padding) ----
    log_probs = torch.log(pitch_probs.clamp(min=1e-8))  # (B, L, C)

    if attention_mask is not None:
        # Flatten and mask out padding positions
        log_probs_flat = log_probs.view(-1, C)  # (B*L, C)
        targets_flat = target_pitch_tokens.view(-1)  # (B*L,)
        valid_flat = attention_mask.view(-1).bool()  # (B*L,)

        if valid_flat.any():
            l_ce = F.nll_loss(
                log_probs_flat[valid_flat],
                targets_flat[valid_flat],
                reduction="mean",
            )
        else:
            # All padding: return zero loss
            l_ce = pitch_probs.sum() * 0.0
    else:
        l_ce = F.nll_loss(
            log_probs.view(-1, C),
            target_pitch_tokens.view(-1),
            reduction="mean",
        )

    # ---- L_seg: Euclidean distance based segment boundary loss (eq.(7)) ----
    if L > 1:
        diff = pitch_probs[:, 1:, :] - pitch_probs[:, :-1, :]  # (B, L-1, C)
        dist_sq = (diff ** 2).sum(dim=-1)  # (B, L-1)

        # boundary markers from GT: 1 where pitch changes
        b_t = (target_pitch_tokens[:, 1:] != target_pitch_tokens[:, :-1]).float()  # (B, L-1)

        # non-boundary: penalize large change; boundary: penalize small change
        l_seg_per = (1 - b_t) * dist_sq + b_t * F.relu(delta - dist_sq)

        if attention_mask is not None:
            seg_mask = attention_mask[:, 1:] * attention_mask[:, :-1]  # (B, L-1)
            l_seg = (l_seg_per * seg_mask).sum() / seg_mask.sum().clamp(min=1.0)
        else:
            l_seg = l_seg_per.mean()
    else:
        l_seg = pitch_probs.sum() * 0.0

    # ---- L_dur: soft duration loss (eq.(8)) ----
    # Each note contributes: (sum of predicted prob for correct pitch in the note span - n_frames)^2
    l_dur_total = pitch_probs.sum() * 0.0  # keep graph connected; initialize to zero
    n_notes = 0
    for b in range(B):
        t_start = 0
        for a_d_i, m_p_i in zip(frame_alignment[b], pitch_note_labels[b]):
            if m_p_i == 0:  # unvoiced note, skip
                t_start += a_d_i
                continue
            t_end = min(t_start + a_d_i, L)
            if t_start >= L:
                break
            # sum of predicted probability for the correct pitch class over the note span
            prob_sum = pitch_probs[b, t_start:t_end, m_p_i].sum()
            l_dur_total = l_dur_total + (prob_sum - a_d_i) ** 2
            n_notes += 1
            t_start += a_d_i
    l_dur = l_dur_total / max(n_notes, 1)

    total = l_ce + lambda_seg * l_seg + lambda_dur * l_dur
    return total, {"l_ce": l_ce, "l_seg": l_seg, "l_dur": l_dur}


def compute_total_loss(
    l_scl: torch.Tensor,
    l_fcl: torch.Tensor,
    l_svt: torch.Tensor,
    l_mask: torch.Tensor,
    lambda_cl: float = 0.5,
    lambda_scl: float = 1.0,
    lambda_fcl: float = 0.1,
    lambda_svt: float = 0.5,
    lambda_mask: float = 0.3,
) -> tuple:
    """Aggregate all loss components.

    L_CL    = lambda_scl * L_SCL + lambda_fcl * L_FCL
    L_total = lambda_cl * L_CL + lambda_svt * L_SVT + lambda_mask * L_mask

    Returns:
        (total_loss, {"l_scl", "l_fcl", "l_cl", "l_svt", "l_mask", "l_total"})

    NOTE: lambda values are paper Section IV-B confirmed:
        lambda_cl=0.5, lambda_scl=1.0, lambda_fcl=0.1, lambda_svt=0.5, lambda_mask=0.3
    """
    l_cl = lambda_scl * l_scl + lambda_fcl * l_fcl
    total = lambda_cl * l_cl + lambda_svt * l_svt + lambda_mask * l_mask
    return total, {
        "l_scl": l_scl,
        "l_fcl": l_fcl,
        "l_cl": l_cl,
        "l_svt": l_svt,
        "l_mask": l_mask,
        "l_total": total,
    }
