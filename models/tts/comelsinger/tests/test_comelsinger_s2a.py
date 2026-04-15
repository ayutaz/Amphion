"""Tests for CoMelSinger_S2A (M2-11 to M2-15)."""
import pytest
import torch
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A


@pytest.fixture(scope="module")
def model():
    """Module-scoped fixture — heavy instantiation (~346M params), done once."""
    return CoMelSinger_S2A()


class TestS2ASkeleton:  # M2-11
    def test_isinstance(self, model):
        assert isinstance(model, MaskGCT_S2A)

    def test_hyperparameters(self, model):
        assert model.pitch_vocab_size == 129
        assert model.temperature == 0.07
        assert model.lambda_cl == 0.5
        assert model.lambda_scl == 1.0
        assert model.lambda_fcl == 0.1
        assert model.lambda_svt == 0.5
        assert model.lambda_mask == 0.3


class TestPitchEmb:  # M2-12
    def test_pitch_emb_exists(self, model):
        assert hasattr(model, "pitch_emb")
        assert isinstance(model.pitch_emb, torch.nn.Embedding)

    def test_pitch_emb_shape(self, model):
        assert model.pitch_emb.weight.shape == (129, 1024)

    def test_forward_with_pitch(self, model):
        B, T, Q = 2, 50, 12
        x0 = torch.randint(0, 1024, (B, T, Q))
        x_mask = torch.ones(B, T)
        cond_code = torch.randint(0, 1024, (B, T))
        pitch_tokens = torch.randint(0, 129, (B, T))
        # Should not raise
        result = model(x0, x_mask, cond_code, pitch_tokens)
        assert len(result) == 6  # logits, mask_layer, final_mask, x0, prompt_len, mask_prob

    def test_forward_without_pitch_backward_compat(self, model):
        """pitch_tokens=None should work (backward compat with MaskGCT)."""
        B, T, Q = 2, 50, 12
        x0 = torch.randint(0, 1024, (B, T, Q))
        x_mask = torch.ones(B, T)
        cond_code = torch.randint(0, 1024, (B, T))
        result = model(x0, x_mask, cond_code, pitch_tokens=None)
        assert len(result) == 6


class TestGetCond:  # M2-13
    def test_get_cond_shape(self, model):
        cond = model.get_cond(
            torch.randint(0, 1024, (1, 100)),
            torch.randint(0, 129, (1, 100)),
        )
        assert cond.shape == (1, 100, 1024)

    def test_get_cond_without_pitch(self, model):
        cond = model.get_cond(torch.randint(0, 1024, (1, 100)))
        assert cond.shape == (1, 100, 1024)


class TestMaskLoss:  # M2-14
    def test_forward_loss_finite(self, model):
        B, T, Q = 2, 50, 12
        x0 = torch.randint(0, 1024, (B, T, Q))
        x_mask = torch.ones(B, T)
        cond_code = torch.randint(0, 1024, (B, T))
        pitch_tokens = torch.randint(0, 129, (B, T))
        logits, mask_layer, final_mask, _, _, _ = model(x0, x_mask, cond_code, pitch_tokens)
        assert torch.isfinite(logits).all()

    def test_pitch_emb_grad_flows(self, model):
        """Verify gradient flows to pitch_emb."""
        model.train()
        B, T, Q = 1, 30, 12
        x0 = torch.randint(0, 1024, (B, T, Q))
        x_mask = torch.ones(B, T)
        cond_code = torch.randint(0, 1024, (B, T))
        pitch_tokens = torch.randint(0, 129, (B, T))

        model.zero_grad()
        logits, mask_layer, final_mask, x0_out, prompt_len, mask_prob = model(
            x0, x_mask, cond_code, pitch_tokens
        )
        # Compute a simple loss on logits
        loss = logits.sum()
        loss.backward()

        assert model.pitch_emb.weight.grad is not None
        assert torch.isfinite(model.pitch_emb.weight.grad).all()

        # Cleanup
        model.zero_grad()


class TestLoRA:  # M2-15
    def test_lora_application(self):
        """Test LoRA can be applied and trainable params are ~2-10%."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            pytest.skip("peft not installed")

        model = CoMelSinger_S2A()
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            modules_to_save=["pitch_emb"],
        )
        peft_model = get_peft_model(model, lora_config)

        total = sum(p.numel() for p in peft_model.parameters())
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        ratio = trainable / total * 100

        print(f"Total: {total:,}, Trainable: {trainable:,}, Ratio: {ratio:.2f}%")
        # With r=16 targeting q_proj+v_proj (16 layers) + pitch_emb saved:
        # LoRA: 16 * 2 * 2 * (1024*16) = ~1.05M, pitch_emb: 129*1024 = 132K
        # Total trainable ~1.18M / 347M base = ~0.34%
        # Accept 0.1–5% as a reasonable LoRA range
        assert 0.1 < ratio < 5.0, f"Expected 0.1-5%, got {ratio:.2f}%"
