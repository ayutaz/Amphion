"""Tests for CoMelSinger loss functions (M1-07 to M1-11)."""
import pytest
import torch
import torch.nn.functional as F

from models.tts.comelsinger.losses import (
    compute_fcl_loss,
    compute_scl_loss,
    compute_svt_loss,
    compute_total_loss,
)

SEED = 42


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# TestSCLLoss
# ---------------------------------------------------------------------------


class TestSCLLoss:
    def test_identical_inputs_near_zero(self):
        """g_a == g_b -> loss < 1e-4 (NT-Xent with identical embeddings)."""
        g = torch.randn(8, 512)
        loss = compute_scl_loss(g, g)
        assert loss.item() < 1e-4

    def test_random_inputs_positive(self):
        """Random distinct embeddings produce positive loss."""
        g_a = torch.randn(8, 512)
        g_b = torch.randn(8, 512)
        loss = compute_scl_loss(g_a, g_b)
        assert loss.item() > 0

    def test_higher_temperature_lower_loss(self):
        """Higher tau (softer distribution) produces lower loss than low tau."""
        g_a = torch.randn(8, 512)
        g_b = torch.randn(8, 512)
        loss_small_tau = compute_scl_loss(g_a, g_b, tau=0.07)
        loss_large_tau = compute_scl_loss(g_a, g_b, tau=1.0)
        # Smaller tau sharpens the distribution -> larger loss for hard negatives
        assert loss_small_tau.item() > loss_large_tau.item()

    def test_minimal_k_s(self):
        """K_s=1 should not raise errors."""
        g_a = torch.randn(1, 128)
        g_b = torch.randn(1, 128)
        loss = compute_scl_loss(g_a, g_b)
        assert torch.isfinite(loss)

    def test_output_is_scalar(self):
        """Return value must be a 0-dimensional scalar tensor."""
        g_a = torch.randn(4, 256)
        g_b = torch.randn(4, 256)
        loss = compute_scl_loss(g_a, g_b)
        assert loss.ndim == 0

    def test_backward_propagates(self):
        """Loss must be differentiable w.r.t. both inputs."""
        g_a = torch.randn(8, 512, requires_grad=True)
        g_b = torch.randn(8, 512, requires_grad=True)
        loss = compute_scl_loss(g_a, g_b)
        loss.backward()
        assert g_a.grad is not None
        assert g_b.grad is not None


# ---------------------------------------------------------------------------
# TestFCLLoss
# ---------------------------------------------------------------------------


class TestFCLLoss:
    def test_zero_Y_returns_zero(self):
        """All-zero Y matrix -> loss exactly 0.0."""
        B, L, D = 2, 10, 64
        f_a = torch.randn(B, L, D)
        f_b = torch.randn(B, L, D)
        Y = torch.zeros(B, L, L)
        loss = compute_fcl_loss(f_a, f_b, Y)
        assert loss.item() == 0.0

    def test_random_positive(self):
        """Random inputs with non-zero Y produce finite positive loss."""
        B, L, D = 2, 10, 64
        f_a = torch.randn(B, L, D)
        f_b = torch.randn(B, L, D)
        # Identity matrix: frame i matches only frame i
        Y = torch.eye(L).unsqueeze(0).expand(B, -1, -1).contiguous()
        loss = compute_fcl_loss(f_a, f_b, Y)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_minimal_sequence(self):
        """L=1 should not raise errors."""
        B, L, D = 2, 1, 32
        f_a = torch.randn(B, L, D)
        f_b = torch.randn(B, L, D)
        Y = torch.ones(B, L, L)
        loss = compute_fcl_loss(f_a, f_b, Y)
        assert torch.isfinite(loss)

    def test_output_is_scalar(self):
        """Return value must be a 0-dimensional scalar tensor."""
        B, L, D = 2, 5, 32
        f_a = torch.randn(B, L, D)
        f_b = torch.randn(B, L, D)
        Y = torch.eye(L).unsqueeze(0).expand(B, -1, -1).contiguous()
        loss = compute_fcl_loss(f_a, f_b, Y)
        assert loss.ndim == 0

    def test_backward_propagates(self):
        """Loss must be differentiable w.r.t. both frame embeddings."""
        B, L, D = 2, 8, 64
        f_a = torch.randn(B, L, D, requires_grad=True)
        f_b = torch.randn(B, L, D, requires_grad=True)
        Y = torch.eye(L).unsqueeze(0).expand(B, -1, -1).contiguous()
        loss = compute_fcl_loss(f_a, f_b, Y)
        loss.backward()
        assert f_a.grad is not None
        assert f_b.grad is not None


# ---------------------------------------------------------------------------
# TestSVTLoss
# ---------------------------------------------------------------------------


def _make_svt_inputs(B=2, L=20, C=129, n_notes=4):
    """Helper: create minimal valid SVT loss inputs."""
    logits = torch.randn(B, L, C)
    pitch_probs = F.softmax(logits, dim=-1)
    target_pitch_tokens = torch.randint(1, C, (B, L))

    # Distribute L frames evenly across n_notes
    frames_per_note = L // n_notes
    remainder = L - frames_per_note * n_notes
    frame_alignment = []
    pitch_note_labels = []
    for _ in range(B):
        fa = [frames_per_note] * n_notes
        if remainder > 0:
            fa[-1] += remainder
        frame_alignment.append(fa)
        pitch_note_labels.append([torch.randint(1, C, (1,)).item() for _ in range(n_notes)])

    attention_mask = torch.ones(B, L)
    return pitch_probs, target_pitch_tokens, frame_alignment, pitch_note_labels, attention_mask


class TestSVTLoss:
    def test_perfect_prediction_low_dur(self):
        """When frame counts match exactly, l_dur should be near zero."""
        B, L, C = 2, 20, 129
        n_notes = 4
        frames_per_note = L // n_notes  # 5 frames each

        # Build frame_alignment and pitch_note_labels
        frame_alignment = [[frames_per_note] * n_notes for _ in range(B)]
        pitch_labels = [torch.randint(1, C, (n_notes,)).tolist() for _ in range(B)]

        # Construct pitch_probs such that for each note, probs[t, m_p_i] = 1.0
        # This means prob_sum = frames_per_note exactly -> l_dur = 0
        pitch_probs = torch.zeros(B, L, C)
        for b in range(B):
            t = 0
            for i, (a_d, m_p) in enumerate(zip(frame_alignment[b], pitch_labels[b])):
                pitch_probs[b, t : t + a_d, m_p] = 1.0
                t += a_d
        pitch_probs.requires_grad_(True)

        target_pitch_tokens = torch.zeros(B, L, dtype=torch.long)
        for b in range(B):
            t = 0
            for a_d, m_p in zip(frame_alignment[b], pitch_labels[b]):
                target_pitch_tokens[b, t : t + a_d] = m_p
                t += a_d

        attention_mask = torch.ones(B, L)

        total, parts = compute_svt_loss(
            pitch_probs,
            target_pitch_tokens,
            frame_alignment,
            pitch_labels,
            attention_mask=attention_mask,
        )
        assert parts["l_dur"].item() < 1e-4, f"l_dur={parts['l_dur'].item()}"

    def test_zero_lambda_excludes(self):
        """lambda_seg=0, lambda_dur=0 -> total equals l_ce."""
        pitch_probs, target_pitch_tokens, fa, pl, am = _make_svt_inputs()
        pitch_probs = pitch_probs.detach().requires_grad_(True)

        total, parts = compute_svt_loss(
            pitch_probs,
            target_pitch_tokens,
            fa,
            pl,
            attention_mask=am,
            lambda_seg=0.0,
            lambda_dur=0.0,
        )
        assert abs(total.item() - parts["l_ce"].item()) < 1e-5

    def test_return_dict_keys(self):
        """Returned dict must contain exactly l_ce, l_seg, l_dur."""
        pitch_probs, target_pitch_tokens, fa, pl, am = _make_svt_inputs()
        _, parts = compute_svt_loss(pitch_probs, target_pitch_tokens, fa, pl, attention_mask=am)
        assert set(parts.keys()) == {"l_ce", "l_seg", "l_dur"}

    def test_backward_propagates(self):
        """total must be differentiable w.r.t. pitch_probs."""
        pitch_probs, target_pitch_tokens, fa, pl, am = _make_svt_inputs()
        logits = torch.randn_like(pitch_probs)
        logits.requires_grad_(True)
        pitch_probs = F.softmax(logits, dim=-1)

        total, _ = compute_svt_loss(pitch_probs, target_pitch_tokens, fa, pl, attention_mask=am)
        total.backward()
        assert logits.grad is not None

    def test_all_padding(self):
        """All-padding attention_mask -> l_ce is finite (not NaN)."""
        B, L, C = 2, 20, 129
        pitch_probs, target_pitch_tokens, fa, pl, _ = _make_svt_inputs(B=B, L=L, C=C)
        all_padding = torch.zeros(B, L)  # 0 = masked out everywhere

        total, parts = compute_svt_loss(
            pitch_probs,
            target_pitch_tokens,
            fa,
            pl,
            attention_mask=all_padding,
        )
        assert torch.isfinite(parts["l_ce"]), f"l_ce is NaN/Inf: {parts['l_ce']}"
        assert torch.isfinite(total), f"total is NaN/Inf: {total}"


# ---------------------------------------------------------------------------
# TestTotalLoss
# ---------------------------------------------------------------------------


class TestTotalLoss:
    def test_all_zero(self):
        """All-zero inputs -> total == 0."""
        z = torch.tensor(0.0)
        total, _ = compute_total_loss(z, z, z, z)
        assert total.item() == 0.0

    def test_default_lambda_formula(self):
        """Verify formula with paper Section IV-B confirmed lambda values."""
        l_scl = torch.tensor(1.0)
        l_fcl = torch.tensor(2.0)
        l_svt = torch.tensor(3.0)
        l_mask = torch.tensor(4.0)

        # L_CL   = 1.0 * l_scl + 0.1 * l_fcl = 1.0 + 0.2 = 1.2
        # L_total = 0.5 * 1.2 + 0.5 * 3.0 + 0.3 * 4.0 = 0.6 + 1.5 + 1.2 = 3.3
        expected = 0.5 * (1.0 * 1.0 + 0.1 * 2.0) + 0.5 * 3.0 + 0.3 * 4.0
        total, _ = compute_total_loss(l_scl, l_fcl, l_svt, l_mask)
        assert abs(total.item() - expected) < 1e-6, f"got {total.item()}, expected {expected}"

    def test_zero_lambda_mask(self):
        """lambda_mask=0 -> mask component contributes nothing."""
        l_scl = torch.tensor(1.0)
        l_fcl = torch.tensor(1.0)
        l_svt = torch.tensor(1.0)
        l_mask = torch.tensor(999.0)  # large value, should be excluded

        total_with, _ = compute_total_loss(l_scl, l_fcl, l_svt, l_mask, lambda_mask=0.0)
        total_zero, _ = compute_total_loss(l_scl, l_fcl, l_svt, torch.tensor(0.0), lambda_mask=0.0)
        assert abs(total_with.item() - total_zero.item()) < 1e-6

    def test_return_dict_keys(self):
        """Returned dict must contain all 6 expected keys."""
        z = torch.tensor(0.0)
        _, parts = compute_total_loss(z, z, z, z)
        expected_keys = {"l_scl", "l_fcl", "l_cl", "l_svt", "l_mask", "l_total"}
        assert set(parts.keys()) == expected_keys

    def test_backward_propagates(self):
        """total must be differentiable w.r.t. all 4 loss inputs."""
        l_scl = torch.tensor(1.0, requires_grad=True)
        l_fcl = torch.tensor(1.0, requires_grad=True)
        l_svt = torch.tensor(1.0, requires_grad=True)
        l_mask = torch.tensor(1.0, requires_grad=True)

        total, _ = compute_total_loss(l_scl, l_fcl, l_svt, l_mask)
        total.backward()

        assert l_scl.grad is not None
        assert l_fcl.grad is not None
        assert l_svt.grad is not None
        assert l_mask.grad is not None

    def test_custom_lambda(self):
        """Custom lambda values produce the correct aggregated total."""
        l_scl = torch.tensor(2.0)
        l_fcl = torch.tensor(3.0)
        l_svt = torch.tensor(4.0)
        l_mask = torch.tensor(5.0)

        lscl, lfcl, lcl, lsvt, lmask = 0.2, 0.3, 0.4, 0.6, 0.8
        # L_CL = 0.2 * 2.0 + 0.3 * 3.0 = 0.4 + 0.9 = 1.3
        # L_total = 0.4 * 1.3 + 0.6 * 4.0 + 0.8 * 5.0 = 0.52 + 2.4 + 4.0 = 6.92
        expected = lcl * (lscl * 2.0 + lfcl * 3.0) + lsvt * 4.0 + lmask * 5.0
        total, _ = compute_total_loss(
            l_scl, l_fcl, l_svt, l_mask,
            lambda_cl=lcl,
            lambda_scl=lscl,
            lambda_fcl=lfcl,
            lambda_svt=lsvt,
            lambda_mask=lmask,
        )
        assert abs(total.item() - expected) < 1e-5, f"got {total.item()}, expected {expected}"
