# CoMelSinger 再現実装 要件定義書

> **対象論文**: "CoMelSinger: Discrete Token-Based Zero-Shot Singing Synthesis With Structured Melody Control and Guidance" (Zhao et al., 2026, arXiv:2509.19883v2)
> **作成日**: 2026-04-15
> **作成方法**: 論文原文(全14p)精読 + MaskGCTソースコード解析 + Amphionフレームワーク構造調査 + 関連技術調査（6エージェント並列）
> **前提**: 要求定義書 `11_requirements_definition.md` を基に、実装可能な技術仕様へ具体化

---

## 重要: 要求定義からの修正事項

論文原文(Section IV-B)とMaskGCTソースコード解析により、要求定義書から以下を修正する。

| 項目 | 要求定義の値 | **論文原文/コード確認値** | 根拠 |
|---|---|---|---|
| λ_SCL | 0.5 | **1.0** | 論文 Section IV-B |
| λ_FCL | 1.0 | **0.1** | 論文 Section IV-B |
| λ_SVT | 0.1 | **0.5** | 論文 Section IV-B |
| λ_CL | (未記載) | **0.5** | 論文 Section IV-B |
| λ_mask | 0.3 | **0.3** (変更なし) | 論文 Section IV-B |
| SVT学習率 | 1e-4 | **1e-5** | 論文 Section IV-B |
| SVT学習量 | 1000エポック | **50Kステップ** | 論文 Section IV-B |
| RVQコードブック数 | 8 | **12** | MaskGCT `num_quantizer=12`, 論文SVT節 |
| 音響コーデック | `facebook/encodec_24khz` | **Amphion独自 `CodecEncoder/CodecDecoder`** | `maskgct_utils.py` |
| K_s (SCL用) | K//2 (推定) | **8** (バッチ32中) | 論文 Section IV-B |
| S2A学習スケジューラ | コサインアニーリング | **逆平方根スケジューラ** | 論文 Section IV-B |
| 公式実装 | https://github.com/ishine/CoMelSinger | **未公開 (404)** | Web確認 2026-04-15 |

---

## 目次

1. [システム全体設計](#1-システム全体設計)
2. [共通基盤](#2-共通基盤)
3. [RQ-01: ピッチトークナイザー](#3-rq-01-ピッチトークナイザー)
4. [RQ-02: SVTモジュール](#4-rq-02-svtモジュール)
5. [RQ-03: CoMelSinger_S2A](#5-rq-03-comelsinger_s2a)
6. [RQ-04: 損失関数群](#6-rq-04-損失関数群)
7. [RQ-05: データ前処理パイプライン](#7-rq-05-データ前処理パイプライン)
8. [RQ-06: Dataset/DataLoader](#8-rq-06-datasetdataloader)
9. [RQ-07: 学習パイプライン](#9-rq-07-学習パイプライン)
10. [RQ-08: 推論パイプライン](#10-rq-08-推論パイプライン)
11. [RQ-09: 評価システム](#11-rq-09-評価システム)
12. [設定スキーマ](#12-設定スキーマ)
13. [実装順序と依存関係](#13-実装順序と依存関係)

---

## 1. システム全体設計

### 1.1 MaskGCTとの統合アーキテクチャ

MaskGCTソースコード解析に基づく正確なデータフロー:

```
[学習時]
x0: (B, T, 12)     音響トークン（12層RVQ）
x_mask: (B, T)      パディングマスク
cond_code: (B, T)   セマンティックトークン

                    ┌─────────────────────────────────┐
cond_code ─────────►│ cond_emb (nn.Embedding 1024→1024)│
                    └──────────┬──────────────────────┘
                               │ cond: (B, T, 1024)
pitch_tokens ─────►│ pitch_emb (nn.Embedding 129→1024)│
                    └──────────┬──────────────────────┘
                               │ pitch: (B, T, 1024)
                               ▼
                    cond = cond + pitch    ← ★ element-wise加算
                               │
                    cond += layer_emb(mask_layer)  ← loss_t()内
                               │
                    cond_embedding = cond_mlp(cond)  ← DiffLlama内
                               │
                    x += cond_embedding    ← ★ hidden statesに直接加算
                               │
                    ┌──────────▼──────────────────────┐
                    │ DiffLlama (16層 LlamaDecoder)     │
                    │ AdaptiveRMSNorm(diffusion_step)   │
                    └──────────┬──────────────────────┘
                               │ embeds: (B, T, 1024)
                               ▼
                    logits = to_logits[layer](embeds) → (B, T, 1024)
```

### 1.2 condの処理経路（MaskGCTソースコード確認済み）

**重要な発見**: DiffLlamaはcondを**クロスアテンションではなく直接加算**で利用する。

```python
# DiffLlama.forward() 内（llama_nar.py）
cond_embedding = self.cond_mlp(cond)  # (B,T,1024) → MLP → (B,T,1024)
x = x + cond_embedding               # ★ hidden statesに直接加算
```

AdaptiveRMSNormにはdiffusion_stepのみが渡される（condは渡されない）。

### 1.3 ファイル配置

```
models/tts/comelsinger/
├── __init__.py
├── config/
│   └── comelsinger.json            # Amphion JSON5 設定
├── pitch_tokenizer.py              # RQ-01
├── svt_module.py                   # RQ-02
├── comelsinger_s2a.py              # RQ-03
├── losses.py                       # RQ-04
├── preprocess.py                   # RQ-05
├── dataset.py                      # RQ-06
├── train_svt.py                    # RQ-07a
├── train_s2a.py                    # RQ-07b
├── comelsinger_inference.py        # RQ-08
├── evaluate.py                     # RQ-09
└── docs/
```

---

## 2. 共通基盤

### 2.1 Amphion CodecEncoder/Decoder の取得

MaskGCTはAmphion独自のCodecEncoder/Decoderを使用（標準EnCodecではない）。

```python
# maskgct_utils.py での使用パターン
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
```

**対応**: sparse-checkoutに `models/codec` を追加する。

```bash
git sparse-checkout add models/codec
```

### 2.2 RVQコードブック数

MaskGCTのデフォルト: `num_quantizer=12`（論文のSVT節でも「12 discrete acoustic codes」と記載）。

CoMelSingerの全モジュールで `num_quantizer=12` を統一する。

### 2.3 フレームレート

| 系列 | フレームレート | ストライド |
|---|---|---|
| 音響トークン (Amphion Codec) | 75 Hz | 320 samples @ 24kHz |
| セマンティックトークン (w2v-bert-2.0) | 50 Hz | 320 samples @ 16kHz |
| F0 (pyworld DIO, frame_period=5ms) | 200 Hz | 120 samples @ 24kHz |

---

## 3. RQ-01: ピッチトークナイザー

### 3.1 仕様

```python
class PitchTokenizer:
    """F0連続値またはMIDI楽譜からフレームアラインド離散ピッチトークンを生成"""

    VOCAB_SIZE = 129       # 0: unvoiced, 1-128: MIDI note 1-128
    ENCODEC_FPS = 75.0
    F0_FPS = 200.0         # pyworld DIO frame_period=5ms

    def __init__(self, vocab_size=129, encodec_fps=75.0, f0_fps=200.0): ...

    def quantize_f0(self, f0: np.ndarray, target_len: int) -> torch.LongTensor:
        """
        F0 (T_f0,) Hz → ピッチトークン (L,)
        - 各EnCodecフレーム区間のF0中央値でピッチ決定
        - MIDI = round(12 * log2(f/440) + 69), clamp(1,128)
        - 無声(f0==0) → token 0
        """

    def tokenize_score(self, note_midi: List[int], duration_symbol: List[int],
                       target_len: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        楽譜 → (ピッチトークン (L,), フレーム数列 (S,))
        - デュレーション正規化: a_d[i] = floor(m_d[i] / D * L), 端数補正
        - 保証: sum(a_d) == target_len, 各 a_d[i] >= 1
        """

    def compute_soft_label_matrix(self, m_p: torch.LongTensor) -> torch.FloatTensor:
        """
        (L,) → (L,L) ソフトラベル行列
        - Y[i,j]=+1: 同一ピッチ & 両方有声
        - Y[i,j]=0: パディング or 無声
        - 無声同士は正例としない
        """
```

### 3.2 テスト基準

- `quantize_f0(np.array([440.0]), 1)` → `tensor([69])`
- `quantize_f0(np.array([0.0]), 1)` → `tensor([0])`
- `tokenize_score` の `sum(a_d) == target_len` を全ケースで検証
- ソフトラベル行列の対称性検証

---

## 4. RQ-02: SVTモジュール

### 4.1 アーキテクチャ仕様

```python
class SVTModule(nn.Module):
    """
    Encoder-only Transformer: 音響トークン → フレームレベルピッチ予測
    S2A学習時はfrozenで補助監督信号として使用
    """

    def __init__(
        self,
        num_codebooks: int = 12,        # ★ 12 (MaskGCT準拠)
        codebook_size: int = 1024,
        codebook_embed_dim: int = 64,   # 各コードブック埋め込み次元
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        pitch_vocab_size: int = 129,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 各コードブック独立Embedding → 連結 → 線形投影
        self.codebook_embs = nn.ModuleList([
            nn.Embedding(codebook_size, codebook_embed_dim)
            for _ in range(num_codebooks)
        ])
        self.input_proj = nn.Linear(num_codebooks * codebook_embed_dim, hidden_size)
        self.pos_enc = SinusoidalPosEmb(hidden_size)  # llama_nar.pyから流用

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pitch_head = nn.Linear(hidden_size, pitch_vocab_size)
```

### 4.2 入出力テンソル形状

```
入力: acoustic_tokens (B, L, 12) long
       attention_mask  (B, L) bool

内部処理:
  各コードブックEmbed → (B, L, 12, 64) → reshape → (B, L, 768)
  → input_proj → (B, L, 512) → + pos_enc → Transformer
  → pitch_head → (B, L, 129)

出力: {"logits": (B,L,129), "probs": (B,L,129)}
```

### 4.3 損失関数（論文式7,8,9）

```python
def compute_svt_loss(
    pitch_probs,              # (B, L, C) softmax後
    target_pitch_tokens,      # (B, L) GT
    frame_alignment,          # List[List[int]] a^d_i
    pitch_note_labels,        # List[List[int]] m^p_i
    lambda_seg=3.0, lambda_dur=5.0, delta=0.5,
):
    # L_CE: CrossEntropy (padding除外)
    # L_seg: Σ [(1-b_t)*||p_t-p_{t-1}||² + b_t*max(0, δ-||p_t-p_{t-1}||²)]
    # L_dur: Σ_i (Σ_{t=T_i}^{T_i+a^d_i-1} p_t[m^p_i] - a^d_i)²
    return total_loss, {"l_ce": ..., "l_seg": ..., "l_dur": ...}
```

### 4.4 学習仕様（論文原文確認済み）

| 項目 | 値 |
|---|---|
| オプティマイザ | AdamW |
| 学習率 | **1e-5** |
| スケジューラ | コサインアニーリング |
| バッチサイズ | 32 |
| 総ステップ数 | **50,000** |
| 重み減衰 | 0.01 |
| GPU | RTX A5000 × 1 |
| 精度 | FP32 |

---

## 5. RQ-03: CoMelSinger_S2A

### 5.1 継承元の正確なAPI（ソースコード確認済み）

```python
# MaskGCT_S2A の正確なシグネチャ
class MaskGCT_S2A(nn.Module):
    def __init__(self,
        num_quantizer=12,           # ★ デフォルト12
        hidden_size=1024,
        num_layers=16,
        num_heads=16,
        codebook_size=1024,
        cfg_scale=0.15,
        mask_layer_schedule="linear",
        cond_codebook_size=1024,
        cond_dim=1024,
        predict_layer_1=True,
        cfg=None,
    ): ...

    def forward(self, x0, x_mask, cond_code=None):
        # 1. cond = self.cond_emb(cond_code)  → (B,T,1024)
        # 2. self.compute_loss(x0, x_mask, cond) → タプル
        return logits, mask_layer, final_mask, x0, prompt_len, mask_prob

    def compute_loss(self, x0, x_mask, cond=None):
        # t = torch.rand(B).clamp(1e-5, 1.0)
        return self.loss_t(x0, x_mask, t, cond)

    def loss_t(self, x0, x_mask, t, cond=None):
        # 1. xt, new_t, mask_layer, mask, prompt_len, mask_prob = self.forward_diffusion(x0, t)
        # 2. cond += self.layer_emb(mask_layer).unsqueeze(1)  ← ★ インプレース変更
        # 3. embeds = self.diff_estimator(xt, new_t, cond, x_mask)
        # 4. logits = self.to_logits[mask_layer.item()](embeds)
        return logits, mask_layer, final_mask, x0, prompt_len, mask_prob

    @torch.no_grad()
    def reverse_diffusion(self, cond, prompt, x_mask=None, prompt_mask=None,
                          temp=1.5, filter_thres=0.98, max_layer=None,
                          gt_code=None,
                          n_timesteps=[10,4,4,4,4,4,4,4],
                          cfg=1.0, rescale_cfg=1.0):
        # cond: (B, T_full, 1024) — 外部でcond_emb()済みのテンソル
        # prompt: (B, prompt_len, num_quantizer) — プロンプト音響コード
        ...
```

### 5.2 CoMelSinger_S2A 設計

```python
class CoMelSinger_S2A(MaskGCT_S2A):
    def __init__(self, pitch_vocab_size=129, temperature=0.07, **kwargs):
        kwargs.setdefault("num_quantizer", 12)
        super().__init__(**kwargs)

        # ★ 新規追加: ピッチ埋め込み層
        self.pitch_emb = nn.Embedding(pitch_vocab_size, self.hidden_size)

        # ハイパーパラメータ（論文原文確認値）
        self.temperature = temperature
        self.lambda_cl = 0.5
        self.lambda_scl = 1.0     # ★ 修正: 論文原文 1.0
        self.lambda_fcl = 0.1     # ★ 修正: 論文原文 0.1
        self.lambda_svt = 0.5     # ★ 修正: 論文原文 0.5
        self.lambda_mask = 0.3

    def forward(self, x0, x_mask, cond_code=None, pitch_tokens=None):
        """学習用forward — ピッチ埋め込みを追加"""
        cond = self.cond_emb(cond_code)          # (B, T, 1024)
        if pitch_tokens is not None:
            cond = cond + self.pitch_emb(pitch_tokens)  # ★ element-wise加算
        return self.compute_loss(x0, x_mask, cond)

    # ★ 注意: loss_t()はオーバーライド不要
    # loss_t()内でcond += layer_emb()がインプレースで実行されるが、
    # forwardから渡されるcondには既にpitch_embが含まれている
```

### 5.3 推論時のcond構成

```python
# MaskGCT推論パイプラインでは、condは外部で構成してreverse_diffusionに渡す
# （maskgct_utils.py line 198, 210）

# CoMelSinger推論:
cond = model.cond_emb(semantic_code)     # (1, T_total, 1024)
cond = cond + model.pitch_emb(pitch_aligned)  # ★ ピッチ追加
predict = model.reverse_diffusion(
    cond=cond, prompt=acoustic_prompt, ...
)
```

### 5.4 2モデル構成

MaskGCT推論は `s2a_model_1layer` と `s2a_model_full` の2インスタンス構成。

```python
# 推論パイプラインの構成
s2a_1layer = CoMelSinger_S2A(predict_layer_1=True, ...)
s2a_full   = CoMelSinger_S2A(predict_layer_1=False, ...)

# 両方にLoRAとpitch_embを適用
# → 2つのモデルは別インスタンスだがLoRA設定は同一
```

### 5.5 LoRA適用（学習スクリプト側で実行）

```python
# train_s2a.py 内で適用（クラス__init__内ではない）
from peft import LoraConfig, get_peft_model

model = CoMelSinger_S2A(...)
model.load_state_dict(pretrained_maskgct_s2a, strict=False)

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # LlamaAttentionのnn.Linear名
    lora_dropout=0.05, bias="none",
    modules_to_save=["pitch_emb"],  # 通常学習対象
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # → ~4.83%
```

**DDP注意点**: frozen SVTが原因で `find_unused_parameters=True` が必要。

---

## 6. RQ-04: 損失関数群

### 6.1 SCL（シーケンスレベル対照学習）

```python
def compute_scl_loss(g_a, g_b, tau=0.07):
    """
    NT-Xent対称版
    g_a, g_b: (K_s, D) — K_s=8 (論文確認値)
    """
    g_a = F.normalize(g_a, dim=-1)
    g_b = F.normalize(g_b, dim=-1)
    sim_ab = torch.mm(g_a, g_b.T) / tau  # (K_s, K_s)
    sim_ba = torch.mm(g_b, g_a.T) / tau
    labels = torch.arange(g_a.size(0), device=g_a.device)
    return (F.cross_entropy(sim_ab, labels) + F.cross_entropy(sim_ba, labels)) / 2
```

### 6.2 FCL（フレームレベル対照学習）

```python
def compute_fcl_loss(f_a, f_b, Y, tau=0.07):
    """
    ソフトラベル対照学習
    f_a, f_b: (B, L, D)
    Y: (B, L, L) — {+1, -1, 0}
    """
    f_a = F.normalize(f_a, dim=-1)
    f_b = F.normalize(f_b, dim=-1)
    S = torch.bmm(f_a, f_b.transpose(1, 2)) / tau  # (B, L, L)
    valid = (Y != 0).float()
    n_valid = valid.sum().clamp(min=1.0)
    return -(valid * Y * S).sum() / n_valid
```

### 6.3 統合損失（論文原文確認値）

```python
# Algorithm 1 line 23（論文 Section IV-B 確認値）
L_CL   = 1.0 * L_SCL + 0.1 * L_FCL      # λ_SCL=1.0, λ_FCL=0.1
L_SVT  = L_CE + 3 * L_seg + 5 * L_dur    # λ_seg=3, λ_dur=5
L_total = 0.5 * L_CL + 0.5 * L_SVT + 0.3 * L_mask  # λ_CL=0.5, λ_SVT=0.5, λ_mask=0.3
```

---

## 7. RQ-05: データ前処理パイプライン

### 7.1 音響トークン抽出（Amphion Codec）

```python
# sparse-checkout追加が必要: git sparse-checkout add models/codec
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder

# encode（maskgct_utils.py パターンに準拠）
vq_emb = codec_encoder(speech.unsqueeze(1))          # (B,1,T) → 潜在
_, vq, _, _, _ = codec_decoder.quantizer(vq_emb)
acoustic_code = vq.permute(1, 2, 0)                  # (12,B,T) → (B,T,12)

# decode
vq_emb = codec_decoder.vq2emb(codes.permute(2,0,1), n_quantizers=12)
audio = codec_decoder(vq_emb)                         # (B,1,T_wav)
```

### 7.2 セマンティックトークン抽出

```python
# 16kHzにリサンプリング必須
from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor

processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")

inputs = processor(speech_16k, sampling_rate=16000, return_tensors="pt")
outputs = model(**inputs, output_hidden_states=True)
feat = outputs.hidden_states[17]  # 第17層 (B, T_s, 1024)
feat = (feat - semantic_mean) / semantic_std
semantic_code, _ = semantic_codec.quantize(feat)  # (B, T_s)
```

### 7.3 F0抽出とピッチトークン量子化

```python
import pyworld as pw

f0, t = pw.dio(wav_f64, sr=24000, f0_floor=65.0, f0_ceil=1047.0, frame_period=5.0)
f0 = pw.stonemask(wav_f64, f0, t, sr=24000)  # 精緻化

# 200Hz → 75Hzリサンプリング
pitch_tokens = pitch_tokenizer.quantize_f0(f0, target_len=T_a)
```

### 7.4 出力形式

```python
# 各サンプルの.ptファイル
torch.save({
    "acoustic_tokens": tensor(12, T_a),   # ★ 12層
    "semantic_tokens": tensor(T_a,),      # 75Hzにリサンプリング済み
    "pitch_tokens": tensor(T_a,),         # 75Hzにリサンプリング済み
    "phone_ids": tensor(T_ph,),
    "note_durations": tensor(S,),         # 各音符のフレーム数
    "note_pitches": tensor(S,),           # 各音符のピッチトークン
    "attention_mask": tensor(T_a,),
    "speaker_id": int,
}, f"tokens/{uid}.pt")
```

---

## 8. RQ-06: Dataset/DataLoader

### 8.1 バッチ構成（Algorithm 1準拠）

論文 Algorithm 1 (line 2-7) に基づくバッチ構成:

```
バッチB = {x_1, ..., x_K}  (K=32)
  ├── S_A^s = {x_1, ..., x_{K_s}}        K_s=8 サンプル (SCL用)
  └── S_A^f = {x_{K_s+1}, ..., x_K}      24 サンプル (FCL+mask用)

拡張バッチ:
  S_B^f = S_A^f (そのまま)
  S_B^s = P(S_A^f) のピッチ摂動版

  B' = S_B^s ∪ S_B^f にPromptGenでプロンプト付与
```

### 8.2 BalancedSpeakerSampler

SCLの正例ペア構成のため、バッチ内に同一話者の複数サンプルが必要。

```python
class BalancedSpeakerSampler(torch.utils.data.Sampler):
    """K_s=8のSCL用サンプルに同一話者ペアを保証"""
    def __init__(self, speaker_ids, batch_size=32, k_s=8): ...
```

### 8.3 Collate関数

```python
def comelsinger_collate_fn(batch):
    # 可変長パディング (pad_value=0)
    # acoustic_tokens: (B, 12, T_max)  ★ 12層
    # semantic_tokens: (B, T_max)
    # pitch_tokens: (B, T_max)
    # attention_mask: (B, T_max)
    # speaker_id: (B,)
    ...
```

---

## 9. RQ-07: 学習パイプライン

### 9.1 Phase 1: SVT独立学習

```python
# train_svt.py
optimizer = AdamW(svt_model.parameters(), lr=1e-5, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=50000)

for step in range(50000):
    batch = next(dataloader)
    acoustic = batch["acoustic_tokens"]     # (B, T, 12)
    pitch_gt = batch["pitch_tokens"]        # (B, T)

    output = svt_model(acoustic, batch["attention_mask"])
    loss, detail = compute_svt_loss(
        output["probs"], pitch_gt,
        batch["note_durations"], batch["note_pitches"],
        lambda_seg=3.0, lambda_dur=5.0, delta=0.5,
    )
    loss.backward()
    clip_grad_norm_(svt_model.parameters(), 1.0)
    optimizer.step(); scheduler.step()
```

### 9.2 Phase 2: S2Aファインチューニング（Algorithm 1実装）

```python
# train_s2a.py — Algorithm 1の完全実装
accelerator = Accelerator(mixed_precision="bf16",
    kwargs_handlers=[DDPKwargs(find_unused_parameters=True)])

# モデル構築
s2a_model = CoMelSinger_S2A(...)
s2a_model.load_state_dict(pretrained_weights, strict=False)
s2a_model = get_peft_model(s2a_model, lora_config)

svt_model = SVTModule(...)
svt_model.load_state_dict(torch.load("svt_best.pt"))
svt_model.freeze()

optimizer = AdamW(s2a_model.parameters(), lr=1e-5, weight_decay=0.01)

for epoch in range(N):
    for batch in dataloader:
        K = 32; K_s = 8
        # --- Algorithm 1 line 3: バッチ分割 ---
        S_A_s = {k: v[:K_s] for k, v in batch.items()}   # SCL用 8サンプル
        S_A_f = {k: v[K_s:] for k, v in batch.items()}   # FCL+mask用 24サンプル

        # --- line 4: ピッチ摂動 ---
        S_B_s_pitch = pitch_perturbation(S_A_f["pitch_tokens"])

        # --- line 5-6: PromptGen ---
        prompts_a = prompt_gen(batch)
        prompts_b = prompt_gen(batch)  # 異なるプロンプト

        # --- line 9: S2A forward (バッチB) ---
        cond_B = s2a_model.cond_emb(batch["semantic_tokens"])
        cond_B = cond_B + s2a_model.pitch_emb(batch["pitch_tokens"])

        # --- line 10: S2A forward (バッチB') ---
        cond_Bp = s2a_model.cond_emb(batch["semantic_tokens"])
        cond_Bp = cond_Bp + s2a_model.pitch_emb(S_B_s_pitch)

        # --- line 12-13: AvgPool → SCL ---
        g_a = cond_B[:K_s, :prompt_len].mean(dim=1)   # (K_s, D)
        g_b = cond_Bp[:K_s, :prompt_len].mean(dim=1)
        L_SCL = compute_scl_loss(g_a, g_b, tau=0.07)

        # --- line 14-15: フレーム埋め込み → FCL ---
        f_a = cond_B[K_s:]    # (24, T, D)
        f_b = cond_Bp[K_s:]
        Y = build_soft_label_matrix(batch["pitch_tokens"][K_s:])
        L_FCL = compute_fcl_loss(f_a, f_b, Y, tau=0.07)

        # --- line 16: L_CL ---
        L_CL = 1.0 * L_SCL + 0.1 * L_FCL

        # --- line 17-18: マスク予測 ---
        logits, mask_layer, final_mask, x0, _, mask_prob = \
            s2a_model.compute_loss(batch["acoustic_tokens"], batch["attention_mask"], cond_B)
        L_mask = F.cross_entropy(
            logits[final_mask.squeeze(-1).bool()],
            x0[:, :, mask_layer.item()][final_mask.squeeze(-1).bool()]
        )

        # --- line 19-21: SVT損失 (StopGrad) ---
        with torch.no_grad():
            predicted_acoustic = logits.argmax(dim=-1)  # 概念的
            svt_out = svt_model(batch["acoustic_tokens"].detach(),
                               batch["attention_mask"])
        L_SVT, _ = compute_svt_loss(
            svt_out["probs"], batch["pitch_tokens"],
            batch["note_durations"], batch["note_pitches"],
        )

        # --- line 23: 統合損失 ---
        L_total = 0.5 * L_CL + 0.5 * L_SVT + 0.3 * L_mask

        accelerator.backward(L_total)
        clip_grad_norm_(s2a_model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad()
```

---

## 10. RQ-08: 推論パイプライン

### 10.1 MaskGCT推論パイプラインの拡張

```python
class CoMelSinger_Inference_Pipeline(MaskGCT_Inference_Pipeline):
    def __init__(self, ..., pitch_tokenizer, s2a_model_1layer, s2a_model_full, ...):
        super().__init__(
            semantic_model, semantic_codec,
            codec_encoder, codec_decoder,
            t2s_model,
            s2a_model_1layer, s2a_model_full,
            semantic_mean, semantic_std, device,
        )
        self.pitch_tokenizer = pitch_tokenizer

    def comelsinger_inference(
        self, prompt_wav_path, prompt_text,
        lyrics, pitch_sequence, note_durations,
        language="zh", **kwargs,
    ) -> np.ndarray:
        # 1. 前処理
        speech_16k = librosa.load(prompt_wav_path, sr=16000)[0]
        speech_24k = librosa.load(prompt_wav_path, sr=24000)[0]
        pitch_aligned = self.pitch_tokenizer.tokenize_score(
            pitch_sequence, note_durations, target_len
        )[0]

        # 2. T2S
        combine_semantic, _ = self.text2semantic(speech_16k, ...)

        # 3. S2A（ピッチ条件追加）
        acoustic_code = self.extract_acoustic_code(torch.tensor(speech_24k).unsqueeze(0))

        # cond構成: semantic + pitch
        cond = self.s2a_model_1layer.cond_emb(combine_semantic)
        cond = cond + self.s2a_model_1layer.pitch_emb(pitch_aligned)

        # 第1層デコーディング
        predict_1layer = self.s2a_model_1layer.reverse_diffusion(
            cond=cond, prompt=acoustic_code[:,:,:12],
            n_timesteps=[25], cfg=2.5, rescale_cfg=0.75,
        )

        # 全層デコーディング
        cond_full = self.s2a_model_full.cond_emb(combine_semantic)
        cond_full = cond_full + self.s2a_model_full.pitch_emb(pitch_aligned)
        predict_full = self.s2a_model_full.reverse_diffusion(
            cond=cond_full, prompt=acoustic_code[:,:,:12],
            gt_code=predict_1layer,
            n_timesteps=[25,10,1,1,1,1,1,1,1,1,1,1], cfg=2.5, rescale_cfg=0.75,
        )

        # 4. EnCodec Decode
        vq_emb = self.codec_decoder.vq2emb(predict_full.permute(2,0,1), n_quantizers=12)
        audio = self.codec_decoder(vq_emb)[0][0].cpu().numpy()
        return audio
```

---

## 11. RQ-09: 評価システム

### 11.1 指標と論文ターゲット値

| 指標 | Seen目標 | Unseen目標 | 実装 |
|---|---|---|---|
| MOS-Q | 3.90 | 3.87 | 主観評価（20名） |
| MOS-N | 4.02 | 4.11 | 主観評価 |
| SMOS | 4.22 | 4.14 | 主観評価 |
| MCD | 4.17 dB | — | `librosa.feature.mfcc` + DTW |
| F0-RMSE | 0.042 | — | `pyworld` F0抽出 + RMSE |
| SingMOS | 4.32 | 4.25 | https://github.com/South-Twilight/SingMOS |
| SECS | 0.912 | 0.897 | `microsoft/wavlm-base-plus` |
| SVT F1 | 0.711 | — | フレームレベルPrecision/Recall |

### 11.2 Ablation設定（Table V再現用、6条件）

| 構成 | λ_CL | λ_SCL | λ_FCL | λ_SVT |
|---|---|---|---|---|
| Full | 0.5 | 1.0 | 0.1 | 0.5 |
| w/o CL | 0 | — | — | 0.5 |
| w/o SCL | 0.5 | 0 | 0.1 | 0.5 |
| w/o FCL | 0.5 | 1.0 | 0 | 0.5 |
| w/o SVT | 0.5 | 1.0 | 0.1 | 0 |
| w/o CL+SVT | 0 | — | — | 0 |

---

## 12. 設定スキーマ

### 12.1 comelsinger.json（Amphion JSON5形式）

```json5
{
  // Amphion設定継承
  "base_config": "config/base.json",
  "task_type": "svs",

  "model": {
    "s2a": {
      "num_quantizer": 12,
      "hidden_size": 1024,
      "num_layers": 16,
      "num_heads": 16,
      "codebook_size": 1024,
      "cfg_scale": 0.15,
      "cond_codebook_size": 1024,
      "pitch_vocab_size": 129
    },
    "svt": {
      "num_codebooks": 12,
      "codebook_size": 1024,
      "codebook_embed_dim": 64,
      "hidden_size": 512,
      "num_layers": 4,
      "num_heads": 8,
      "pitch_vocab_size": 129,
      "dropout": 0.1
    },
    "lora": {
      "r": 16,
      "lora_alpha": 32,
      "target_modules": ["q_proj", "v_proj"],
      "lora_dropout": 0.05,
      "modules_to_save": ["pitch_emb"]
    }
  },

  "train": {
    "svt": {
      "optimizer": "AdamW",
      "lr": 1e-5,
      "weight_decay": 0.01,
      "max_steps": 50000,
      "batch_size": 32,
      "scheduler": "cosine",
      "gpu_count": 1,
      "precision": "fp32"
    },
    "s2a": {
      "optimizer": "AdamW",
      "lr": 1e-5,
      "weight_decay": 0.01,
      "max_epochs": 100,
      "batch_size": 32,
      "scheduler": "inverse_sqrt",
      "gpu_count": 4,
      "precision": "bf16",
      "k_s": 8
    }
  },

  "loss": {
    "lambda_cl": 0.5,
    "lambda_scl": 1.0,
    "lambda_fcl": 0.1,
    "lambda_svt": 0.5,
    "lambda_mask": 0.3,
    "lambda_seg": 3,
    "lambda_dur": 5,
    "temperature": 0.07,
    "delta": 0.5
  },

  "inference": {
    "n_timesteps_t2s": 50,
    "n_timesteps_s2a": [25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "cfg": 2.5,
    "rescale_cfg": 0.75,
    "temperature": 1.5,
    "filter_thres": 0.98
  }
}
```

---

## 13. 実装順序と依存関係

```
Phase 0: 環境構築
  ├── sparse-checkout に models/codec 追加
  ├── 事前学習済みモデルのダウンロード
  └── 依存ライブラリインストール

Phase 1: 基盤モジュール（並列実装可能）
  ├── [RQ-01] pitch_tokenizer.py   ← 依存なし
  ├── [RQ-04] losses.py            ← 依存なし
  └── [RQ-05] preprocess.py        ← RQ-01に依存

Phase 2: コアモジュール
  ├── [RQ-02] svt_module.py        ← RQ-04に依存
  ├── [RQ-06] dataset.py           ← RQ-05に依存
  └── [RQ-03] comelsinger_s2a.py   ← RQ-04に依存

Phase 3: 学習・推論
  ├── [RQ-07a] train_svt.py        ← RQ-02, RQ-06に依存
  ├── [RQ-07b] train_s2a.py        ← RQ-03, RQ-02(frozen), RQ-06に依存
  ├── [RQ-08] comelsinger_inference.py ← RQ-03に依存
  └── [RQ-09] evaluate.py          ← RQ-08に依存
```

**推定工数**: Phase 1 (3-5日) → Phase 2 (5-7日) → Phase 3 (7-10日)

---

## 付録A: MaskGCT_S2Aメソッド完全リファレンス

### forward_diffusion(x0, t) の戻り値

| フィールド | 形状 | 意味 |
|---|---|---|
| `xt` | `(B, T, hidden_size)` | マスク済み音響埋め込み |
| `new_t` | `(B,)` | 拡散ステップ |
| `mask_layer` | `(1,)` | マスク対象RVQレイヤーインデックス |
| `mask` | `(B, T, 1)` | マスク位置 (1=マスク) |
| `prompt_len` | `(B,)` | プロンプト長（CFGドロップ時0） |
| `mask_prob` | `(B,)` | マスク確率 sin(t*π/2) |

### reverse_diffusion のCFG実装

```python
# maskgct_s2a.py line 411-418
pos_emb_std = embeds.std()
embeds = embeds + cfg * (embeds - mask_embeds)      # g_cfg
rescale_embeds = embeds * pos_emb_std / embeds.std() # g_final
embeds = rescale_cfg * rescale_embeds + (1 - rescale_cfg) * embeds
```

## 付録B: 依存ライブラリ（確認済みバージョン）

```
torch>=2.2.0
torchaudio>=2.2.0
transformers>=4.40.0   # LlamaAttention にq_proj/v_projが存在することを確認
peft>=0.11.0
accelerate>=0.30.0
pyworld>=0.3.4
pypinyin>=0.50.0
librosa>=0.10.1
mir_eval>=0.7
```

## 付録C: 公式実装の状況

- **CoMelSinger**: https://github.com/ishine/CoMelSinger → **404 Not Found** (2026-04-15確認)
- 公式デモサイト: https://danny-nus.github.io/CoMelSinger/ (音声サンプルのみ)
- 論文著者によるコード公開は未実施。再現実装は本プロジェクトが初の試み。
