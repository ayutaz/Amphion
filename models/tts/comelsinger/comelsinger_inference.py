"""CoMelSinger inference pipeline.

Provides singing voice synthesis from score (lyrics + pitch + duration)
and reference audio (timbre).

Two-stage non-autoregressive framework:
  1. T2S: lyrics + pitch -> semantic tokens (via MaskGCT T2S)
  2. S2A: semantic + pitch -> acoustic tokens (via CoMelSinger S2A with pitch conditioning)
  3. Decode: acoustic tokens -> waveform (via Codec decoder)

Usage:
    pipeline = CoMelSingerInferencePipeline.from_pretrained(cfg)
    audio = pipeline.synthesize(
        prompt_wav_path="ref.wav",
        lyrics="ni hao shi jie",
        pitch_sequence=[60, 62, 64, 67],
        note_durations=[0.5, 0.5, 0.5, 0.5],
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CoMelSingerInferencePipeline:
    """Singing voice synthesis inference pipeline.

    Implements the full CoMelSinger inference flow:

    1. Pre-process: tokenize_score -> pitch tokens (frame-aligned)
    2. Extract prompt: reference audio -> semantic + acoustic codes
    3. T2S: text + prompt semantic -> target semantic tokens (MaskGCT T2S)
    4. S2A stage 1: semantic + pitch -> 1st layer acoustic (n_timesteps=[25])
    5. S2A stage 2: semantic + pitch + layer1 -> full 12-layer acoustic
    6. Decode: acoustic tokens -> waveform (Codec decoder)

    All sub-models are optional at construction time; synthesize() raises
    NotImplementedError if required models are missing.
    """

    # Default per-layer diffusion timesteps (12 RVQ layers)
    DEFAULT_N_TIMESTEPS_S2A: List[int] = [25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    def __init__(
        self,
        pitch_tokenizer=None,
        s2a_model_1layer=None,
        s2a_model_full=None,
        t2s_model=None,
        codec_encoder=None,
        codec_decoder=None,
        semantic_model=None,
        semantic_codec=None,
        semantic_mean=None,
        semantic_std=None,
        device: str = "cpu",
    ) -> None:
        """Initialize the inference pipeline.

        Args:
            pitch_tokenizer: PitchTokenizer instance. Created with defaults if None.
            s2a_model_1layer: CoMelSinger_S2A model for 1st RVQ layer prediction.
            s2a_model_full: CoMelSinger_S2A model for full 12-layer prediction.
            t2s_model: MaskGCT T2S model for text-to-semantic generation.
            codec_encoder: Amphion CodecEncoder for prompt acoustic extraction.
            codec_decoder: Amphion CodecDecoder for waveform synthesis.
            semantic_model: Wav2Vec2BertModel for semantic feature extraction.
            semantic_codec: RepCodec for semantic quantization.
            semantic_mean: Precomputed semantic feature mean for normalization.
            semantic_std: Precomputed semantic feature std for normalization.
            device: Computation device string ("cpu", "cuda", "cuda:0", etc.).
        """
        from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer

        self.pitch_tokenizer = pitch_tokenizer or PitchTokenizer()
        self.s2a_model_1layer = s2a_model_1layer
        self.s2a_model_full = s2a_model_full
        self.t2s_model = t2s_model
        self.codec_encoder = codec_encoder
        self.codec_decoder = codec_decoder
        self.semantic_model = semantic_model
        self.semantic_codec = semantic_codec
        self.semantic_mean = semantic_mean
        self.semantic_std = semantic_std
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _check_models_ready(self) -> None:
        """Verify that all required models are loaded."""
        missing = []
        if self.s2a_model_1layer is None:
            missing.append("s2a_model_1layer")
        if self.s2a_model_full is None:
            missing.append("s2a_model_full")
        if self.t2s_model is None:
            missing.append("t2s_model")
        if self.codec_encoder is None:
            missing.append("codec_encoder")
        if self.codec_decoder is None:
            missing.append("codec_decoder")
        if self.semantic_model is None:
            missing.append("semantic_model")
        if self.semantic_codec is None:
            missing.append("semantic_codec")
        if missing:
            raise NotImplementedError(
                f"Full inference requires pretrained models. "
                f"Missing: {', '.join(missing)}. "
                f"Use from_pretrained() to load models first."
            )

    @torch.no_grad()
    def synthesize(
        self,
        prompt_wav_path: str | Path,
        lyrics: str,
        pitch_sequence: List[int],
        note_durations: List[float],
        language: str = "zh",
        n_timesteps_s2a: Optional[List[int]] = None,
        cfg_scale: float = 2.5,
        rescale_cfg: float = 0.75,
    ) -> np.ndarray:
        """Synthesize singing voice from score + reference audio.

        Args:
            prompt_wav_path: Path to reference audio WAV (for timbre extraction).
            lyrics: Lyrics text (Chinese pinyin or characters).
            pitch_sequence: MIDI note numbers per note, shape (S,).
                           e.g. [60, 62, 64, 67] for C4-D4-E4-G4.
            note_durations: Duration in seconds per note, shape (S,).
                           e.g. [0.5, 0.5, 0.5, 0.5].
            language: Language code for G2P ("zh", "en").
            n_timesteps_s2a: Per-layer diffusion timesteps, length 12.
                            Defaults to [25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1].
            cfg_scale: Classifier-free guidance scale.
            rescale_cfg: CFG rescaling factor.

        Returns:
            audio: Numpy array (T,) at 24kHz sample rate.

        Raises:
            NotImplementedError: If required pretrained models are not loaded.
            FileNotFoundError: If prompt_wav_path does not exist.
        """
        self._check_models_ready()

        if n_timesteps_s2a is None:
            n_timesteps_s2a = list(self.DEFAULT_N_TIMESTEPS_S2A)

        prompt_wav_path = Path(prompt_wav_path)
        if not prompt_wav_path.exists():
            raise FileNotFoundError(
                f"Reference audio not found: {prompt_wav_path}"
            )

        # Step 1: Score -> pitch tokens (frame-aligned)
        pitch_tokens, frame_counts = self.tokenize_score(
            pitch_sequence, note_durations
        )
        pitch_aligned = pitch_tokens.unsqueeze(0).to(self.device)  # (1, T)

        # Step 2: Extract prompt features
        prompt_semantic, prompt_acoustic = self.extract_prompt(prompt_wav_path)

        # Step 3: T2S — generate target semantic tokens
        target_len = pitch_tokens.shape[0]
        combine_semantic = self.run_t2s(
            prompt_semantic=prompt_semantic,
            lyrics=lyrics,
            language=language,
            target_len=target_len,
            cfg=cfg_scale,
            rescale_cfg=rescale_cfg,
        )

        # Step 4-5: S2A — generate acoustic tokens with pitch conditioning
        recovered_audio = self.run_s2a(
            combine_semantic=combine_semantic,
            prompt_acoustic=prompt_acoustic,
            pitch_aligned=pitch_aligned,
            n_timesteps=n_timesteps_s2a,
            cfg=cfg_scale,
            rescale_cfg=rescale_cfg,
        )

        return recovered_audio

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def tokenize_score(
        self,
        pitch_sequence: List[int],
        note_durations: List[float],
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        """Convert score to frame-aligned pitch tokens.

        Args:
            pitch_sequence: MIDI note numbers (S,).
            note_durations: Duration in seconds per note (S,).

        Returns:
            pitch_tokens: Frame-aligned pitch tokens (T,).
            frame_counts: Number of frames per note (S,).
        """
        total_dur = sum(note_durations)
        target_len = max(1, int(total_dur * self.pitch_tokenizer.encodec_fps))

        # Convert seconds to relative integer durations (ms-based)
        duration_ms = [max(1, int(d * 1000)) for d in note_durations]

        pitch_tokens, frame_counts = self.pitch_tokenizer.tokenize_score(
            pitch_sequence,
            duration_ms,
            target_len,
        )
        return pitch_tokens, frame_counts

    @torch.no_grad()
    def extract_prompt(
        self,
        wav_path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract semantic and acoustic codes from reference audio.

        Args:
            wav_path: Path to reference WAV file.

        Returns:
            semantic_code: (1, T_prompt) semantic token IDs.
            acoustic_code: (1, T_prompt, num_quantizer) acoustic token IDs.
        """
        import librosa

        # Load at 16kHz for semantic extraction
        speech_16k = librosa.load(str(wav_path), sr=16000)[0]
        # Load at 24kHz for acoustic extraction
        speech_24k = librosa.load(str(wav_path), sr=24000)[0]

        # Semantic features via w2v-bert-2.0
        try:
            from transformers import SeamlessM4TFeatureExtractor

            processor = SeamlessM4TFeatureExtractor.from_pretrained(
                "facebook/w2v-bert-2.0"
            )
            inputs = processor(speech_16k, sampling_rate=16000, return_tensors="pt")
            input_features = inputs["input_features"][0].unsqueeze(0).to(self.device)
            attention_mask = inputs["attention_mask"][0].unsqueeze(0).to(self.device)
        except ImportError:
            raise ImportError(
                "transformers package required for semantic extraction. "
                "Install with: pip install transformers"
            )

        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (1, T, C)
        feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)
        semantic_code, _ = self.semantic_codec.quantize(feat)  # (1, T)

        # Acoustic codes via Amphion Codec
        speech_tensor = torch.tensor(speech_24k).unsqueeze(0).to(self.device)
        vq_emb_acoustic = self.codec_encoder(speech_tensor.unsqueeze(1))
        _, vq, _, _, _ = self.codec_decoder.quantizer(vq_emb_acoustic)
        acoustic_code = vq.permute(1, 2, 0)  # (1, T, num_quantizer)

        return semantic_code, acoustic_code

    @torch.no_grad()
    def run_t2s(
        self,
        prompt_semantic: torch.Tensor,
        lyrics: str,
        language: str,
        target_len: int,
        n_timesteps: int = 50,
        cfg: float = 2.5,
        rescale_cfg: float = 0.75,
    ) -> torch.Tensor:
        """Run Text-to-Semantic stage.

        For CoMelSinger, this reuses MaskGCT T2S directly (no modification).

        Args:
            prompt_semantic: (1, T_prompt) prompt semantic codes.
            lyrics: Input lyrics text.
            language: Language code for G2P.
            target_len: Target semantic token length.
            n_timesteps: Number of diffusion steps for T2S.
            cfg: Classifier-free guidance scale.
            rescale_cfg: CFG rescaling factor.

        Returns:
            combine_semantic: (1, T_prompt + T_target) concatenated semantic codes.
        """
        from models.tts.maskgct.maskgct_utils import g2p_

        phone_ids = g2p_(lyrics, language)[1]
        phone_id = torch.tensor(phone_ids, dtype=torch.long).to(self.device)

        predict_semantic = self.t2s_model.reverse_diffusion(
            prompt_semantic[:, :],
            target_len,
            phone_id.unsqueeze(0),
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
        )

        combine_semantic = torch.cat(
            [prompt_semantic[:, :], predict_semantic], dim=-1
        )
        return combine_semantic

    @torch.no_grad()
    def run_s2a(
        self,
        combine_semantic: torch.Tensor,
        prompt_acoustic: torch.Tensor,
        pitch_aligned: torch.Tensor,
        n_timesteps: List[int],
        cfg: float = 2.5,
        rescale_cfg: float = 0.75,
        temperature: float = 1.5,
        filter_thres: float = 0.98,
    ) -> np.ndarray:
        """Run Semantic-to-Acoustic stage with pitch conditioning.

        Two-pass generation:
        1. 1-layer model predicts first RVQ layer from semantic + pitch
        2. Full model predicts remaining layers conditioned on first layer

        Args:
            combine_semantic: (1, T_total) combined semantic codes.
            prompt_acoustic: (1, T_prompt, num_quantizer) prompt acoustic codes.
            pitch_aligned: (1, T_target) frame-aligned pitch tokens.
            n_timesteps: Per-layer timesteps, length 12.
            cfg: Classifier-free guidance scale.
            rescale_cfg: CFG rescaling factor.
            temperature: Sampling temperature.
            filter_thres: Top-k filtering threshold.

        Returns:
            audio: (T,) numpy array at 24kHz.
        """
        # Build condition with pitch embedding (CoMelSinger extension)
        cond_1layer = self.s2a_model_1layer.get_cond(
            combine_semantic, pitch_aligned
        )
        prompt = prompt_acoustic[:, :, :]

        # Stage 1: predict 1st RVQ layer
        predict_1layer = self.s2a_model_1layer.reverse_diffusion(
            cond=cond_1layer,
            prompt=prompt,
            temp=temperature,
            filter_thres=filter_thres,
            n_timesteps=n_timesteps[:1],
            cfg=cfg,
            rescale_cfg=rescale_cfg,
        )

        # Stage 2: predict all 12 layers conditioned on 1st layer
        cond_full = self.s2a_model_full.get_cond(
            combine_semantic, pitch_aligned
        )
        predict_full = self.s2a_model_full.reverse_diffusion(
            cond=cond_full,
            prompt=prompt,
            temp=temperature,
            filter_thres=filter_thres,
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
            gt_code=predict_1layer,
        )

        # Decode to waveform
        vq_emb = self.codec_decoder.vq2emb(
            predict_full.permute(2, 0, 1), n_quantizers=12
        )
        recovered_audio = self.codec_decoder(vq_emb)
        recovered_audio = recovered_audio[0][0].cpu().numpy()

        return recovered_audio

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        ckpt_dir: str | Path,
        lora_path: Optional[str | Path] = None,
        device: str = "cpu",
    ) -> "CoMelSingerInferencePipeline":
        """Load all models from pretrained checkpoints.

        Loads:
        - Semantic model (w2v-bert-2.0) + normalization stats
        - Semantic codec (RepCodec)
        - Acoustic codec (CodecEncoder + CodecDecoder)
        - T2S model (MaskGCT T2S)
        - S2A models (CoMelSinger S2A 1-layer + full, with optional LoRA)
        - PitchTokenizer

        Args:
            ckpt_dir: Directory containing amphion_maskgct/ subfolder
                      (e.g., 'models/tts/maskgct/ckpt').
            lora_path: Optional path to LoRA adapter weights directory.
            device: Computation device string ("cpu", "cuda", "mps").

        Returns:
            Initialized pipeline with all models loaded.
        """
        import safetensors.torch
        from safetensors import safe_open

        from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
        from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer
        from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S
        from models.tts.maskgct.maskgct_utils import (
            build_acoustic_codec,
            build_semantic_codec,
        )
        from transformers import Wav2Vec2BertModel
        from utils.util import load_config

        ckpt_dir = Path(ckpt_dir)
        amphion_dir = ckpt_dir / "amphion_maskgct"
        cfg_path = ckpt_dir.parent / "config" / "maskgct.json"

        # 1. Load MaskGCT config
        cfg = load_config(str(cfg_path))

        # 2. Build and load acoustic codec (encoder + decoder)
        codec_encoder, codec_decoder = build_acoustic_codec(
            cfg.model.acoustic_codec, device
        )
        safetensors.torch.load_model(
            codec_encoder,
            str(amphion_dir / "acoustic_codec" / "model.safetensors"),
        )
        # CodecDecoder: shared tensor workaround via safe_open
        dec_weights: dict = {}
        with safe_open(
            str(amphion_dir / "acoustic_codec" / "model_1.safetensors"),
            framework="pt",
        ) as f:
            for k in f.keys():
                dec_weights[k] = f.get_tensor(k)
        codec_decoder.load_state_dict(dec_weights, strict=False)

        # 3. Build and load semantic codec (RepCodec)
        semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
        safetensors.torch.load_model(
            semantic_codec,
            str(amphion_dir / "semantic_codec" / "model.safetensors"),
        )

        # 4. Build and load T2S model
        t2s_model = MaskGCT_T2S(
            hidden_size=1536,
            num_layers=16,
            num_heads=16,
            cfg_scale=0.15,
            cond_codebook_size=8192,
            cond_dim=1024,
        )
        skip_keys = {"diff_estimator.embed_tokens.weight"}
        t2s_weights: dict = {}
        with safe_open(
            str(amphion_dir / "t2s_model" / "model.safetensors"),
            framework="pt",
        ) as f:
            for k in f.keys():
                if k not in skip_keys:
                    t2s_weights[k] = f.get_tensor(k)
        t2s_model.load_state_dict(t2s_weights, strict=False)

        # 5. Build and load S2A models as CoMelSinger_S2A (1-layer + full)
        s2a_kwargs = dict(
            cond_codebook_size=8192,
            cond_dim=1024,
            hidden_size=1024,
            num_layers=16,
            num_heads=16,
            codebook_size=1024,
            cfg_scale=0.15,
        )

        s2a_1layer = CoMelSinger_S2A(
            num_quantizer=1, predict_layer_1=True, **s2a_kwargs
        )
        s2a_weights_1l: dict = {}
        with safe_open(
            str(amphion_dir / "s2a_model" / "s2a_model_1layer" / "model.safetensors"),
            framework="pt",
        ) as f:
            for k in f.keys():
                if k not in skip_keys:
                    s2a_weights_1l[k] = f.get_tensor(k)
        s2a_1layer.load_state_dict(s2a_weights_1l, strict=False)

        s2a_full = CoMelSinger_S2A(
            num_quantizer=12, predict_layer_1=False, **s2a_kwargs
        )
        s2a_weights_full: dict = {}
        with safe_open(
            str(amphion_dir / "s2a_model" / "s2a_model_full" / "model.safetensors"),
            framework="pt",
        ) as f:
            for k in f.keys():
                if k not in skip_keys:
                    s2a_weights_full[k] = f.get_tensor(k)
        s2a_full.load_state_dict(s2a_weights_full, strict=False)

        # 6. Load w2v-bert-2.0 semantic model
        semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")

        # 7. Load normalization stats
        stats = torch.load(
            str(ckpt_dir / "wav2vec2bert_stats.pt"),
            map_location="cpu",
            weights_only=True,
        )

        # 8. Move all models to device and set eval mode
        dev = torch.device(device)
        codec_encoder.to(dev).eval()
        codec_decoder.to(dev).eval()
        semantic_codec.to(dev).eval()
        t2s_model.to(dev).eval()
        s2a_1layer.to(dev).eval()
        s2a_full.to(dev).eval()
        semantic_model.to(dev).eval()

        # 9. Build pipeline instance
        pipeline = cls(
            pitch_tokenizer=PitchTokenizer(),
            s2a_model_1layer=s2a_1layer,
            s2a_model_full=s2a_full,
            t2s_model=t2s_model,
            codec_encoder=codec_encoder,
            codec_decoder=codec_decoder,
            semantic_model=semantic_model,
            semantic_codec=semantic_codec,
            semantic_mean=stats["mean"].to(dev),
            semantic_std=stats["var"].sqrt().to(dev),
            device=device,
        )

        # 10. Optionally load LoRA weights
        if lora_path is not None:
            pipeline.load_lora_weights(Path(lora_path))

        return pipeline

    def load_lora_weights(
        self,
        lora_path: str | Path,
        model_type: str = "both",
    ) -> None:
        """Load LoRA adapter weights for S2A models.

        After training, LoRA adapters are saved separately. This method
        wraps the base S2A models with PEFT LoRA adapters.

        Args:
            lora_path: Directory containing LoRA adapter subdirectories:
                       lora_path/s2a_1layer/ and lora_path/s2a_full/.
            model_type: Which model(s) to load LoRA for.
                       "1layer" - only 1-layer model
                       "full" - only full model
                       "both" - both models (default)

        Raises:
            ImportError: If peft package is not installed.
            ValueError: If model_type is not one of "1layer", "full", "both".
        """
        if model_type not in ("1layer", "full", "both"):
            raise ValueError(
                f"model_type must be '1layer', 'full', or 'both', got '{model_type}'"
            )

        lora_path = Path(lora_path)

        # Early return if no models need loading
        needs_1layer = model_type in ("1layer", "both") and self.s2a_model_1layer is not None
        needs_full = model_type in ("full", "both") and self.s2a_model_full is not None
        if not needs_1layer and not needs_full:
            logger.info("No S2A models to load LoRA weights for (models are None)")
            return

        from peft import PeftModel

        if needs_1layer:
            adapter_path = lora_path / "s2a_1layer"
            logger.info("Loading LoRA weights for s2a_1layer from %s", adapter_path)
            self.s2a_model_1layer = PeftModel.from_pretrained(
                self.s2a_model_1layer,
                str(adapter_path),
            )

        if needs_full:
            adapter_path = lora_path / "s2a_full"
            logger.info("Loading LoRA weights for s2a_full from %s", adapter_path)
            self.s2a_model_full = PeftModel.from_pretrained(
                self.s2a_model_full,
                str(adapter_path),
            )
