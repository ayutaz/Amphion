# CoMelSinger 再現実装 要求定義書

> **対象論文**: "CoMelSinger: Discrete Token-Based Zero-Shot Singing Synthesis With Structured Melody Control and Guidance" (Zhao et al., 2026, arXiv:2509.19883v2)
> **作成日**: 2026-04-15
> **作成方法**: 論文分析ドキュメント10本 + MaskGCTソースコード調査に基づく6エージェント並列分析

---

## 目次

1. [システム概要](#1-システム概要)
2. [モジュール要求定義](#2-モジュール要求定義)
   - 2.1 [ピッチトークナイザー](#21-ピッチトークナイザー)
   - 2.2 [SVTモジュール](#22-svtモジュール)
   - 2.3 [S2A拡張（CoMelSinger_S2A）](#23-s2a拡張comelsinger_s2a)
3. [データ前処理パイプライン](#3-データ前処理パイプライン)
4. [学習パイプライン](#4-学習パイプライン)
5. [推論パイプライン](#5-推論パイプライン)
6. [評価システム](#6-評価システム)
7. [未確定事項・要確認項目](#7-未確定事項要確認項目)

---

## 1. システム概要

### 1.1 全体アーキテクチャ

```
楽譜(歌詞+ピッチ) + 参照音声(音色)
        │
        ▼
┌─────────────────────────────────────┐
│  Stage 1: T2S (Text-to-Semantic)    │  MaskGCT ベース（そのまま転用）
│  歌詞+ピッチ → セマンティックトークン   │
└───────────────┬─────────────────────┘
                ▼
┌─────────────────────────────────────┐
│  Stage 2: S2A (Semantic-to-Acoustic)│  ★ CoMelSinger拡張
│  + ピッチ埋め込み条件付け             │
│  + Contrastive Learning (SCL+FCL)   │
│  + SVT ピッチ監督（frozen）          │
└───────────────┬─────────────────────┘
                ▼
┌─────────────────────────────────────┐
│  EnCodec Decoder（事前学習済み）      │
│  音響トークン(RVQ 8層) → 波形         │
└─────────────────────────────────────┘
```

### 1.2 実装モジュール一覧

| モジュール | ファイル | 優先度 | 依存 |
|---|---|---|---|
| ピッチトークナイザー | `pitch_tokenizer.py` | 高 | なし |
| SVTモジュール | `svt_module.py` | 高 | ピッチトークナイザー |
| S2A拡張 | `comelsinger_s2a.py` | 高 | SVT, ピッチトークナイザー |
| 対照学習損失 | `losses.py` | 高 | なし |
| データ前処理 | `preprocess.py` | 高 | ピッチトークナイザー |
| Datasetクラス | `dataset.py` | 中 | 前処理済みデータ |
| 学習スクリプト | `train_svt.py`, `train_s2a.py` | 中 | 全モジュール |
| 推論パイプライン | `comelsinger_inference.py` | 中 | S2A拡張 |
| 評価スクリプト | `evaluate.py` | 低 | 推論パイプライン |

### 1.3 ファイル配置

```
models/tts/comelsinger/
├── __init__.py
├── pitch_tokenizer.py       # ピッチトークナイザー
├── svt_module.py             # SVT (Singing Voice Transcription) モジュール
├── comelsinger_s2a.py        # S2A拡張クラス（MaskGCT_S2A継承）
├── losses.py                 # 損失関数（SCL, FCL, SVT損失, 統合損失）
├── dataset.py                # Dataset / DataLoader / Sampler
├── preprocess.py             # データ前処理パイプライン
├── train_svt.py              # SVT独立学習スクリプト
├── train_s2a.py              # S2A fine-tuningスクリプト
├── comelsinger_inference.py  # 推論パイプライン
├── evaluate.py               # 評価スクリプト
├── configs/                  # ハイパーパラメータ設定
│   ├── svt_train.yaml
│   ├── s2a_train.yaml
│   └── inference.yaml
└── docs/                     # ドキュメント（既存）
    ├── 01_overview.md ... 10_implementation_requirements.md
    └── 11_requirements_definition.md  # 本ドキュメント
```

---

## 2. モジュール要求定義

---

### 2.1 ピッチトークナイザー

**ファイル**: `models/tts/comelsinger/pitch_tokenizer.py`
**依存**: `torch`, `numpy`, `math` のみ（F0抽出器は呼び出し側が担当）

#### 2.1.1 機能要件

**2つの入力モード**

| モード | 用途 | 入力 | 出力 |
|---|---|---|---|
| F0量子化 | 学習前処理・GT生成 | F0連続値列 `(T_f0,)` Hz | ピッチトークン列 `(L,)` |
| 楽譜変換 | 推論・学習入力 | MIDIノート番号 + デュレーション | ピッチトークン列 `(L,)` + フレーム数列 `(S,)` |

**量子化方式（等分平均律 MIDI量子化）**

```python
# 有声フレーム: F0 (Hz) → MIDI番号 → トークン
n = round(12 * log2(f / 440.0) + 69)
n = clamp(n, 1, 128)
token = n  # インデックス 1–128

# 無声フレーム: f == 0 or NaN
token = 0  # インデックス 0（無声専用）
```

**ピッチ語彙サイズ**: `C = 129`（0: 無声, 1–128: MIDIノート番号1–128）

**フレームレート変換（F0 → EnCodecフレームレート）**

- F0フレームレート: 200Hz（pyworld DIO想定）
- EnCodecフレームレート: 75Hz（24kHz / stride 320）
- 変換方法: 各EnCodecフレーム区間の有声F0中央値でピッチを決定

**デュレーション正規化（論文式(2)）**

```python
# シンボリックデュレーション m_d → フレーム割り当て a_d
D = sum(m_d)
a_d = [floor(m_d_i / D * L) for i in range(S)]
# 端数補正: sum(a_d) == L を保証（最長音符から1フレームずつ加算）
```

**ソフトラベル行列生成（FCL用）**

```python
# Y_f[i, j] = 1.0 if m_p[i] == m_p[j] and both voiced, else 0.0
```

#### 2.1.2 インターフェース仕様

```python
class PitchTokenizer:
    def __init__(
        self,
        vocab_size: int = 129,
        encodec_fps: float = 75.0,
        f0_fps: float = 200.0,
        unvoiced_interp: bool = False,
    ) -> None: ...

    def quantize_f0(
        self,
        f0: np.ndarray,                    # (T_f0,) Hz, 0.0=unvoiced
        target_len: Optional[int] = None,   # EnCodecフレーム数L
    ) -> torch.LongTensor: ...              # (L,)

    def tokenize_score(
        self,
        note_midi: List[int],               # (S,) MIDIノート番号（0=休符）
        duration_symbol: List[int],          # (S,) シンボリックデュレーション
        target_len: int,                     # EnCodecフレーム数L
    ) -> Tuple[torch.LongTensor, torch.LongTensor]: ...  # (L,), (S,)

    def compute_soft_label_matrix(
        self,
        m_p: torch.LongTensor,             # (L,)
    ) -> torch.FloatTensor: ...             # (L, L)

    def decode_token_to_freq(
        self, token: int,
    ) -> float: ...                         # Hz（token==0 → 0.0）
```

#### 2.1.3 テスト基準

- A4 = 440Hz → token 69 に変換されること
- 無声フレーム（f0=0.0, NaN）→ token 0
- `sum(a_d) == target_len` が常に成立すること
- 各 `a_d[i] >= 1`（0フレームの音符が発生しないこと）
- ソフトラベル行列 `Y_f` が対称であること
- 無声トークン同士を正例としないこと（`Y_f[i,j]=0` when both unvoiced）

---

### 2.2 SVTモジュール

**ファイル**: `models/tts/comelsinger/svt_module.py`
**依存**: `torch.nn`, ピッチトークナイザー

#### 2.2.1 アーキテクチャ仕様

| パラメータ | 値 |
|---|---|
| 構造 | Encoder-only Transformer（双方向フルアテンション） |
| レイヤー数 | 4 |
| 隠れ次元 | 512 |
| アテンションヘッド数 | 8（1ヘッドあたり64次元） |
| FFN中間次元 | 2048（hidden_size × 4） |
| ドロップアウト | 0.1（独立学習時のみ） |
| 位置エンコーディング | SinusoidalPosEmb（MaskGCTと同方式） |
| 正規化 | 標準LayerNorm（AdaptiveRMSNormは不要） |

**入力**: 8層RVQアコースティックトークン → 各層Embedding → 連結 → 線形投影(→512) → 位置エンコーディング

**出力**: フレームレベルピッチ分類logits `(B, L, C)`

#### 2.2.2 損失関数（3成分）

```
L_SVT = L_CE + λ_seg * L_seg + λ_dur * L_dur
        λ_seg = 3, λ_dur = 5
```

**(a) L_CE: クロスエントロピー損失**

```
L_CE = -1/L_valid * Σ_{t: valid} log(p_t[m^p_t])
```

**(b) L_seg: セグメント遷移損失（論文式7）**

```
L_seg = Σ_{t=2}^{L} [(1 - b_t) * ||p_t - p_{t-1}||^2 + b_t * max(0, δ - ||p_t - p_{t-1}||^2)]
```
- `b_t`: ピッチ境界マーカー（`gt_pitch[t] != gt_pitch[t-1]` で1）
- `δ`: マージン（推奨: 0.5〜1.0、実験で調整）

**(c) L_dur: ソフトデュレーション損失（論文式8）**

```
L_dur = Σ_{i=1}^{S} (Σ_{t=T_i}^{T_i + a^d_i - 1} p_t[m^p_i] - a^d_i)^2
```
- `a^d_i`: i番目の音符のフレーム数
- `p_t[m^p_i]`: フレームtでの正解ピッチクラスの予測確率

#### 2.2.3 StopGrad操作（S2A学習時）

```python
# S2A fine-tuning時
svt_module.freeze()  # requires_grad_(False)
with torch.no_grad():
    pitch_logits = svt_module(acoustic_tokens.detach())
# L_SVTはS2Aパラメータへの間接的誘導信号として使用
```

- SVT独立学習時: 学習可能（`requires_grad=True`）
- S2A fine-tuning時: 完全凍結（`requires_grad=False`）
- 推論時: SVTモジュール不使用（ロードもしない）

#### 2.2.4 インターフェース仕様

```python
class SVTModule(nn.Module):
    def __init__(
        self,
        num_codebooks: int = 8,
        codebook_size: int = 1024,
        codebook_embed_dim: int = 64,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        pitch_vocab_size: int = 129,
        dropout: float = 0.1,
    ) -> None: ...

    def forward(
        self,
        acoustic_tokens: torch.LongTensor,              # (B, L, 8)
        attention_mask: Optional[torch.BoolTensor],      # (B, L)
    ) -> dict: ...  # {"logits": (B,L,C), "probs": (B,L,C)}

    def freeze(self) -> None: ...
    def unfreeze(self) -> None: ...
```

```python
def compute_svt_loss(
    pitch_probs: torch.FloatTensor,          # (B, L, C)
    target_pitch_tokens: torch.LongTensor,   # (B, L)
    frame_alignment: List[List[int]],        # a^d
    pitch_note_labels: List[List[int]],      # m^p_i
    lambda_seg: float = 3.0,
    lambda_dur: float = 5.0,
    delta: float = 0.5,
) -> Tuple[torch.Tensor, dict]: ...  # (total_loss, {l_ce, l_seg, l_dur})
```

#### 2.2.5 テスト基準

- フレームレベル評価: Precision, Recall, F1（目標F1: 0.711、M4Singer+Opencpop複合学習時）
- ノートレベル評価: Note F1（`mir_eval.transcription`）
- アブレーション: L_CE only → +L_seg → +L_dur で段階的改善を確認
- S2Aとの接続: frozen SVTからの勾配がS2Aパラメータに流れないことを確認

---

### 2.3 S2A拡張（CoMelSinger_S2A）

**ファイル**: `models/tts/comelsinger/comelsinger_s2a.py`
**継承元**: `models/tts/maskgct/maskgct_s2a.py` の `MaskGCT_S2A`

#### 2.3.1 ピッチ埋め込み拡張

```python
# MaskGCT_S2Aのcond_embに対し、ピッチ埋め込みをelement-wise加算
self.pitch_emb = nn.Embedding(pitch_vocab_size, hidden_size)  # hidden_size=1024

# forward内
cond = self.cond_emb(semantic_tokens) + self.pitch_emb(pitch_tokens)
```

**Length Expansion**: ノート単位ピッチ `(B, S)` → フレーム単位 `(B, L)` への拡張メソッドを実装

#### 2.3.2 対照学習（SCL + FCL）

**(a) SCL（シーケンスレベル対照学習）**

- 目的: グローバルな音色とピッチを分離（prosody leakage防止）
- 正例: 同一バッチ位置のオリジナルvs摂動版ペア
- 負例: 異なるバッチ位置の全ペア
- 類似度: コサイン類似度（L2正規化後の内積）
- 温度: `τ = 0.07`（初期値、設定可能）
- **対称化**: 双方向で計算し平均

```
L_SCL = 1/(2K) * Σ_τ [
    log( exp(<g_τ^a, g_τ^b>/τ) / Σ_r exp(<g_τ^a, g_r^b>/τ) )
  + log( exp(<g_τ^b, g_τ^a>/τ) / Σ_r exp(<g_r^b, g_τ^a>/τ) )
]
```

**(b) FCL（フレームレベル対照学習）**

- 目的: フレームレベルでの音色とピッチの細粒度分離
- ソフトラベル行列 `Y_f[i,j]`: 同一ピッチトークン=+1、異なる=-1、パディング/繰り返し=0
- 繰り返しトークン判定: `α = 0.1` パラメータ

**ピッチ摂動（Pitch Perturbation）**: ランダムピッチシフト（`zero_prob=0.5` で50%の確率でシフトなし）

#### 2.3.3 総合損失関数

```
L_total = λ_CL * L_CL + λ_SVT * L_SVT + λ_mask * L_mask

L_CL   = λ_SCL * L_SCL + λ_FCL * L_FCL
L_SVT  = L_CE + λ_seg * L_seg + λ_dur * L_dur  (frozen SVT経由)
L_mask = MaskGCTマスク予測損失（既存compute_lossを流用）
```

**損失重み一覧**

| 係数 | 値 | 役割 |
|---|---|---|
| `λ_SCL` | 0.5 | シーケンスレベル対照損失 |
| `λ_FCL` | 1.0 | フレームレベル対照損失 |
| `λ_SVT` | 0.1 | SVTピッチ監督（補助信号） |
| `λ_mask` | 0.3 | MaskGCTマスク予測損失 |
| `λ_seg` | 3 | SVT内セグメント遷移損失 |
| `λ_dur` | 5 | SVT内デュレーション損失 |

> **注意**: `03_contrastive_learning.md` と `05_training_inference.md` で `λ_SCL` と `λ_FCL` の値が逆転して記載されている。`05_training_inference.md` の `λ_SCL=0.5, λ_FCL=1.0` がAlgorithm 1と整合するため、こちらを正とする。実装前に論文原文 Section III-D を再確認すること。

#### 2.3.4 LoRA適用仕様

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    modules_to_save=["pitch_emb"],  # ピッチ埋め込みは通常学習
)
```

- 全パラメータに対する更新率: **約4.83%**（論文Table VII）
- 凍結対象: MaskGCT S2Aの元の全重み、SVTモジュール、EnCodecデコーダ
- 学習対象: LoRA差分行列（q_proj, v_proj各層）、ピッチ埋め込み層

#### 2.3.5 インターフェース仕様

```python
class CoMelSinger_S2A(MaskGCT_S2A):
    def __init__(
        self,
        # --- 既存引数 ---
        num_quantizer=8, hidden_size=1024, num_layers=16, num_heads=16,
        codebook_size=1024, cfg_scale=0.15, mask_layer_schedule="linear",
        cond_codebook_size=1024, cond_dim=1024, predict_layer_1=True, cfg=None,
        # --- 新規追加引数 ---
        pitch_vocab_size=129, pad_pitch_id=0, temperature=0.07, alpha=0.1,
        lambda_scl=0.5, lambda_fcl=1.0, lambda_svt=0.1, lambda_mask=0.3,
    ): ...

    def forward(self, x0, x_mask, cond_code=None, pitch_tokens=None): ...

    def expand_pitch_to_frame_level(self, pitch_tokens_note, durations, target_len): ...
    def pitch_perturbation(self, pitch_tokens, zero_prob=0.5): ...
    def compute_scl_loss(self, g_a, g_b): ...
    def build_soft_label_matrix(self, pitch_frame): ...
    def compute_fcl_loss(self, f_a, f_b, pitch_frame): ...
    def compute_s2a_loss(self, x0, x_mask, cond_code, pitch_tokens,
                         pitch_tokens_perturbed, svt_model=None,
                         ground_truth_pitch=None): ...
```

#### 2.3.6 テスト基準

- ピッチ埋め込み: `cond_emb(sem) + pitch_emb(pitch)` の出力shape `(B, T, 1024)` を確認
- Length Expansion: `sum(a_d) == target_len` かつ各音符の正しいトークンが展開されること
- SCL損失: 完全一致ペアで損失≈0、ランダムペアで損失>0
- ソフトラベル行列: 対称性、パディング行列の0化、`{+1, -1, 0}` のみ含むことを確認
- 統合損失: NaN/Inf が発生しないこと、各成分がスカラーであること

---

## 3. データ前処理パイプライン

**ファイル**: `models/tts/comelsinger/preprocess.py`, `dataset.py`

### 3.1 データセット要件

| データセット | 規模 | ライセンス | 用途 |
|---|---|---|---|
| M4Singer | 700曲+, 20話者, ~29h | CC BY-NC-SA 4.0 | S2A学習/評価（主力） |
| Opencpop | 100曲, 1話者, ~5.2h | 研究目的限定 | S2A学習/SVT評価 |
| MIR-ST500 | 500曲, 160k+注釈ノート | 研究目的限定 | SVT汎化評価 |

### 3.2 前処理パイプライン（7ステップ）

```
Step 1: リサンプリング (44.1kHz → 24kHz)
    │
Step 2: EnCodec → 音響トークン (8, T_a) @ 75Hz
    │
Step 3: Wav2Vec2-BERT → セマンティックトークン (T_s,) @ 50Hz
    │                    → 75Hzにリサンプリング
Step 4: F0抽出 (DIO/CREPE) → ピッチトークン量子化 (T_p,) @ 100Hz
    │                        → 75Hzにリサンプリング
Step 5: 歌詞 → ピンイン → 音素ID列 (T_ph,)
    │
Step 6: フレームアライメント（全系列を75Hz基準に統一）
    │
Step 7: セグメント分割・パディング・保存
```

### 3.3 フレームレート整合

| 系列 | 元レート | 目標 | 方法 |
|---|---|---|---|
| 音響トークン | 75 Hz | 75 Hz（基準） | — |
| セマンティックトークン | 50 Hz | 75 Hz | nearest neighborアップサンプリング |
| ピッチトークン | 100 Hz | 75 Hz | 最近傍リサンプリング |
| 音素列 | 音節レベル | フレームレベル | duration展開 |

**長さ検証**: 整合後 `len(acoustic) == len(semantic_75) == len(pitch_75)` を必ず確認

### 3.4 出力データ形式

**保存形式**: PyTorch `.pt` ファイル（個別保存）+ JSON メタデータ

```
preprocessed/
  m4singer/
    tokens/{uid}.pt     # dict: acoustic_tokens(8,T), semantic_tokens(T,),
                        #       pitch_tokens(T,), phone_ids(T_ph,),
                        #       note_durations(T_ph,), attention_mask(T,)
    train.json / val.json / test.json
    singers.json
  opencpop/
    ...
  mir_st500/
    ...
```

### 3.5 Datasetクラス

```python
class CoMelSingerDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_json, max_len=2250, min_len=75): ...
    def __getitem__(self, idx) -> dict: ...  # .ptファイルをロード
```

**対照学習用バッチサンプリング**

```python
class BalancedSpeakerSampler(torch.utils.data.Sampler):
    """
    SCL用にバッチ内で各話者から最低2サンプルを保証。
    バッチ32: 最低4話者 × 8サンプル or 8話者 × 4サンプル
    """
```

**Collator関数**: 可変長シーケンスを同一バッチ内最大長にパディング（`pad_value=0`）

### 3.6 ストレージ見積もり

| 項目 | 容量 |
|---|---|
| 元データ合計（M4Singer + Opencpop + MIR-ST500） | 120〜195 GB |
| 前処理済みトークンキャッシュ | 20〜38 GB |
| MaskGCT事前学習モデル | 5〜15 GB |
| 中間チェックポイント | 10〜30 GB |
| **合計ストレージ** | **約200〜300 GB** |

---

## 4. 学習パイプライン

### 4.1 Phase 1: SVT独立学習

| 項目 | 設定 |
|---|---|
| 目的 | 音響トークン → フレームレベルピッチトークン予測 |
| データ | M4Singer + Opencpop 複合（推奨） |
| GPU | NVIDIA RTX A5000 × 1（24GB） |
| 精度 | FP32 |
| オプティマイザ | AdamW (lr=1e-4, weight_decay=0.01) |
| スケジューラ | コサインアニーリング |
| バッチサイズ | 32 |
| エポック数 | 1000 |
| 損失 | `L_SVT = L_CE + 3 * L_seg + 5 * L_dur` |
| 完了条件 | 1000エポック完走、またはval F1収束 |
| 出力 | `svt_model.pt`（Phase 2で凍結使用） |

### 4.2 Phase 2: S2Aファインチューニング

| 項目 | 設定 |
|---|---|
| 目的 | MaskGCT S2Aをピッチ制御+対照学習で歌唱ドメインに適応 |
| 初期重み | `amphion/MaskGCT-S2A`（HuggingFace） |
| データ | M4Singer + Opencpop 複合 |
| GPU | NVIDIA RTX A5000 × 4（24GB × 4） |
| 精度 | BF16（Mixed Precision） |
| 分散学習 | DDP or HuggingFace Accelerate |
| オプティマイザ | AdamW (lr=1e-5, weight_decay=0.01) |
| スケジューラ | コサインアニーリング |
| バッチサイズ | 32（各GPU 8 × 4GPU） |
| エポック数 | 100 |
| LoRA | r=16, α=32, target=q_proj,v_proj |
| 凍結対象 | MaskGCT S2A元重み、SVT全パラメータ |

**損失関数**

```
L = 0.3 * L_mask + (0.5 * L_SCL + 1.0 * L_FCL) + 0.1 * L_SVT
```

**Algorithm 1 処理順序**

1. バッチ `B` (K=32個) をサンプリング
2. 前半 `K_s` 個 → セグメント用、後半 → フルシーケンス用
3. フルシーケンスにピッチ摂動 `P(·)` を適用 → 拡張バッチ `B'`
4. `PromptGen` で同一話者の参照音声プロンプトを付与
5. S2Aで `B`, `B'` を通し埋め込みを取得
6. AvgPool → シーケンス埋め込み（SCL用）
7. フレーム埋め込み保持（FCL用）
8. `L_SCL`, `L_FCL` → `L_CL` を計算
9. マスク予測 → `L_mask` を計算
10. `StopGrad(â)` → frozen SVT → `L_SVT` を計算
11. 全損失統合 → AdamWで更新

### 4.3 Phase 3: T2Sファインチューニング（オプション）

- 優先度: 低（Phase 1・2完了後に評価結果を見て判断）
- MaskGCT T2Sをそのまま転用する可能性が高い

### 4.4 チェックポイント・ロギング

| Phase | 保存頻度 | ベスト基準 |
|---|---|---|
| Phase 1 (SVT) | 50エポックごと + ベスト | val F1最大 |
| Phase 2 (S2A) | 10エポックごと + ベスト | val F0-RMSE最小 |

**ロギング項目**

- Phase 1: `loss_total`, `loss_ce`, `loss_seg`, `loss_dur`, `val/f1`, `learning_rate`
- Phase 2: `loss_total`, `loss_mask`, `loss_scl`, `loss_fcl`, `loss_svt`, `val/f0_rmse`, `val/secs`, 生成サンプル音声（10エポックごと）

### 4.5 再現性要件

```python
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 4.6 アブレーション実験設定

**Table V（コンポーネント分析）**: 6条件

| 構成 | 変更点 |
|---|---|
| Full | 変更なし |
| w/o CL | λ_SCL=0, λ_FCL=0 |
| w/o SCL | λ_SCL=0 |
| w/o FCL | λ_FCL=0 |
| w/o SVT | λ_SVT=0 |
| w/o CL+SVT | λ_SCL=0, λ_FCL=0, λ_SVT=0 |

**Table VI（SVT訓練条件）**: 5条件のデータ構成比較

**Table VII（Fine-tuning戦略）**: FT-LoRA, FT-LLRD, FT-Prefix, FT-Pets, FT-PIGS, FT-Full

---

## 5. 推論パイプライン

**ファイル**: `models/tts/comelsinger/comelsinger_inference.py`

### 5.1 入力仕様

| パラメータ | 型 | 説明 |
|---|---|---|
| `lyrics` | `str` | 中国語歌詞テキスト |
| `pitch_sequence` | `List[int]` | MIDIノート番号列（0=無声） |
| `note_durations` | `List[float]` | 各音符の持続時間（秒） |
| `prompt_wav_path` | `str` | 参照音声WAVパス（3〜10秒推奨） |
| `prompt_text` | `str` | 参照音声のテキスト |

### 5.2 処理フロー

```
1. 前処理
   ├─ lyrics → pypinyin → phone_id
   ├─ pitch_sequence + note_durations → フレームアライン → m^p_aligned
   └─ prompt_wav → 16kHz (セマンティック用) + 24kHz (音響用)

2. Stage 1: T2S
   ├─ prompt(16kHz) → Wav2Vec2-BERT → セマンティックコード
   └─ phone_id + prompt_semantic → reverse_diffusion(n=50, cfg=2.5)
       → combine_semantic_code

3. Stage 2: S2A（CoMelSinger拡張）
   ├─ cond = cond_emb(semantic) + pitch_emb(m^p_aligned)
   ├─ 第1層: reverse_diffusion(n=25) → predict_1layer
   └─ 全層: reverse_diffusion(n=[25,10,1,...]) → predict_full (8, T_a)

4. EnCodec Decoder
   └─ predict_full → codec_decoder → 歌声波形 (24kHz)
```

### 5.3 推論パラメータ

| パラメータ | デフォルト値 |
|---|---|
| `n_timesteps_t2s` | 50 |
| `n_timesteps_s2a` | [25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] |
| `cfg` | 2.5 |
| `rescale_cfg` | 0.75 |
| `temperature` | 1.5 |
| `filter_thres` | 0.98 |

### 5.4 インターフェース仕様

```python
class CoMelSinger_Inference_Pipeline:
    def __init__(self, semantic_model, semantic_codec, codec_encoder,
                 codec_decoder, t2s_model, s2a_model_1layer,
                 s2a_model_full, pitch_tokenizer, ...): ...

    def comelsinger_inference(
        self,
        prompt_wav_path: str,
        prompt_text: str,
        lyrics: str,
        pitch_sequence: List[int],
        note_durations: List[float],
        language: str = "zh",
        **kwargs,
    ) -> np.ndarray: ...  # (T,) 24kHz歌声波形

    @torch.no_grad()
    def text2semantic(self, ...): ...

    @torch.no_grad()
    def semantic2acoustic(self, ...): ...

    def align_pitch_sequence(self, pitch_sequence, note_durations,
                             target_frame_length, frame_rate=75.0): ...
```

### 5.5 性能要件

| 指標 | 目標値 |
|---|---|
| RTF（単一GPU） | < 0.5 |
| 推論VRAM | 16〜24 GB |
| バッチ推論スループット | ≥ 4サンプル/分 |

---

## 6. 評価システム

**ファイル**: `models/tts/comelsinger/evaluate.py`

### 6.1 評価指標一覧

| 指標 | 種別 | 方向 | 計算方法 |
|---|---|---|---|
| MOS-Q | 主観（品質） | ↑ | 20名評価者、5段階リッカート |
| MOS-N | 主観（自然性） | ↑ | 同上 |
| SMOS | 主観（話者類似度） | ↑ | 同上 |
| MCD | 客観 | ↓ | MFCC(13次元) + DTW距離 (dB) |
| F0-RMSE | 客観 | ↓ | 有声フレームのF0差分RMSE |
| SingMOS | 客観（自動MOS） | ↑ | SingMOSモデル (arXiv:2406.10911) |
| SECS | 客観（話者類似度） | ↑ | WavLM-base+ コサイン類似度 |
| SVT F1 | 客観（ピッチ精度） | ↑ | フレームレベルPrecision/Recall/F1 |

### 6.2 論文ターゲット値

| シナリオ | MOS-Q | MOS-N | SMOS | MCD | F0-RMSE | SingMOS | SECS |
|---|---|---|---|---|---|---|---|
| Seen | 3.90 | 4.02 | 4.22 | 4.17 | 0.042 | 4.32 | 0.912 |
| Unseen | 3.87 | 4.11 | 4.14 | — | — | 4.25 | 0.897 |

### 6.3 評価シナリオ

1. **Seen-singer** (Table II): M4Singerテスト分割、50発話、全指標
2. **Unseen-singer** (Table III): 訓練未見話者、50発話、SingMOS+SECSのみ（GTなし）
3. **Cross-dataset** (Table VI): MIR-ST500でSVT汎化評価
4. **Ablation** (Table IV-VII): 各構成の独立評価

### 6.4 インターフェース仕様

```python
class CoMelSingerEvaluator:
    def compute_mcd(self, synth_wav, ref_wav, sr=24000) -> float: ...
    def compute_f0_rmse(self, synth_wav, ref_wav, sr=24000) -> float: ...
    def compute_secs(self, synth_wav, prompt_wav, sr=24000) -> float: ...
    def compute_singmos(self, synth_wav, sr=24000) -> float: ...
    def evaluate_svt(self, pred_pitch, gt_pitch) -> dict: ...
    def evaluate_batch(self, synth_wavs, ref_wavs, prompt_wavs,
                       scenario="seen") -> dict: ...
```

---

## 7. 未確定事項・要確認項目

### 7.1 論文原文で確認が必要な項目

| 項目 | 現状 | 確認箇所 |
|---|---|---|
| `λ_SCL`, `λ_FCL` の正確な値 | 0.5, 1.0（05ドキュメント準拠） | Section III-D, 式(10)周辺 |
| `λ_mask` の値 | 0.3（05ドキュメント）vs 0.5（06ドキュメント） | Section IV-B |
| `λ_SVT` の値 | 0.1（05ドキュメント）vs 0.5（06ドキュメント） | Section IV-B |
| ウィンドウサイズ12 の意味 | クロマ（12半音）に関連と推定 | Section III-C |
| L_segのマージンδ の値 | 0.5〜1.0（推定） | Section III-C, 式(7) |
| SCL温度パラメータτ | 0.07（推定） | Section III-B |
| ピッチ語彙サイズC の正確な値 | 129（MIDI 1-128 + 無声） | Section III-C |
| S2A内のK_s（対照学習用サンプル数） | K_s = K//2（推定） | Algorithm 1 |

### 7.2 実装判断が必要な項目

| 項目 | 選択肢 | 推奨 |
|---|---|---|
| EnCodecの実装 | Amphion独自(CodecEncoder) vs facebook公式 | 公式 `facebook/encodec_24khz` |
| F0抽出器 | pyworld DIO vs CREPE | pyworld DIO（速度重視） |
| SVT Transformer実装 | `nn.TransformerEncoder` vs 独自実装 | `nn.TransformerEncoder`（シンプル） |
| 前処理キャッシュ形式 | `.pt` 個別 vs HDF5一括 | `.pt` 個別（並列I/O） |
| 分散学習フレームワーク | PyTorch DDP vs Accelerate | Accelerate（Amphionとの親和性） |

### 7.3 ハードウェア互換性

論文はRTX A5000 × 4で実験。利用可能なGPUが異なる場合:
- **24GB VRAM × 1**: SVT学習のみ可能。S2Aはgradient accumulation必須
- **40/80GB VRAM（A100等）**: バッチサイズ増加可能、gradient accumulationで1GPU対応可能
- **16GB VRAM**: LoRAランク削減（r=8）+ gradient checkpointing必須

---

## 付録A: 依存ライブラリ（推奨バージョン）

```
torch==2.2.0
torchaudio==2.2.0
transformers==4.40.0
peft==0.11.0
accelerate==0.30.0
encodec==0.1.1
librosa==0.10.1
soundfile==0.12.1
pyworld==0.3.4
pypinyin==0.50.0
einops==0.7.0
mir_eval==0.7
tensorboard>=2.15.0
```

## 付録B: 事前学習済みモデル

| モデル | 用途 | 入手先 |
|---|---|---|
| MaskGCT (T2S + S2A) | CoMelSingerの初期重み | `amphion/MaskGCT` (HuggingFace) |
| EnCodec 24kHz | 音響トークン化/デコード | `facebook/encodec_24khz` |
| Wav2Vec2-BERT | セマンティック特徴抽出 | `facebook/w2v-bert-2.0` |
| WavLM-base+ | SECS評価 | `microsoft/wavlm-base-plus` |
| SingMOS | 歌唱品質自動評価 | https://github.com/South-Twilight/SingMOS |
