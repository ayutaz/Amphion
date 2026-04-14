# 再現実装に必要な要件と依存関係

CoMelSinger 論文（Zhao et al., 2026, arXiv:2509.19883v2）の再現実装に必要なデータセット・ソフトウェア・ハードウェア・実装コンポーネントを体系的にまとめる。

---

## 1. 必要なデータセット

### 1.1 M4Singer

| 項目 | 内容 |
|------|------|
| 入手先 | GitHub: https://github.com/M4Singer/M4Singer |
| ライセンス | CC BY-NC-SA 4.0（非商用） |
| 言語 | 中国語（普通話） |
| 規模 | 700 曲以上、20 名の歌手、約 29 時間 |
| サンプリングレート | 44.1 kHz（前処理で 24 kHz にダウンサンプリング） |

**コンテンツ**
- 歌声音声ファイル（WAV）
- 歌詞とピッチ注釈（MusicXML / TextGrid 形式）
- ノート・音素レベルのタイムスタンプ
- 各歌手の複数スタイル（ポップ、民謡、オペラなど）

**前処理方法**
1. サンプリングレートを 44.1 kHz から 24 kHz に変換（librosa または soundfile を使用）
2. EnCodec モデル（24 kHz 対応、8 RVQ コードブック）でアコースティックトークンを抽出
3. wav2vec / Whisper 系モデルでセマンティックトークン（フレームレベル）を抽出
4. ground-truth F0（基本周波数）を抽出し、ピッチトークナイザーで離散ピッチトークンに量子化
5. 歌詞テキストをピンイン（拼音）シーケンスに変換
6. 音声の無音区間をトリムし、最大長でセグメント分割
7. 話者 ID を割り当て（M4Singer は 20 話者）

**SVT 学習への利用**
- M4Singer と Opencpop の組み合わせデータセットで SVT（Singing Voice Transcription）モジュールを学習
- ピッチトークン予測の教師信号として ground-truth F0 を使用

---

### 1.2 Opencpop

| 項目 | 内容 |
|------|------|
| 入手先 | 公式サイト: https://wenet.org.cn/opencpop/ （申請制） |
| ライセンス | 研究目的のみ（商用利用不可、再配布不可） |
| 言語 | 中国語（普通話） |
| 規模 | 100 曲、1 名の女性歌手、約 5.2 時間 |
| サンプリングレート | 44.1 kHz |

**コンテンツ**
- 歌声音声ファイル（WAV）
- 詳細なピッチ・音素・ノート境界注釈（TextGrid 形式）
- MIDI 相当のノートシーケンス
- 音素 duration ラベル

**前処理方法**
1. 同上の 44.1 kHz → 24 kHz ダウンサンプリング
2. EnCodec によるアコースティックトークン化（8 コードブック）
3. ピンイン変換と音素アライメントの確認
4. TextGrid からノートレベルおよびフレームレベルの F0 ラベルを抽出
5. ピッチトークナイザーでの量子化（C = ピッチ語彙サイズ でソフトマックス適用）
6. 音声セグメントのトレーニング / バリデーション / テスト分割（論文では seen-singer 評価に使用）

**特記事項**
- Opencpop は注釈品質が高く、SVT モジュールのフレームレベル評価に特に重要
- 論文 Table VI の SVT フレームレベル評価（精度・再現率・F1）は M4Singer / Opencpop / MIR-ST500 の 3 データセットで実施

---

### 1.3 MIR-ST500

| 項目 | 内容 |
|------|------|
| 入手先 | 論文著者提供または GitHub: https://github.com/Jun-CEN/MIR-ST500 |
| ライセンス | 研究目的のみ |
| 言語 | 中国語（500 曲のポップ音楽） |
| 規模 | 500 曲以上、160,000 以上の注釈付きノート |
| 用途 | SVT モジュールの汎化性能評価 |

**SVT 評価での利用方法**
- 事前学習: MIR-ST500 単体でまず SVT を学習 → クロスドメイン汎化を評価
- ファインチューニング: M4Singer + Opencpop の結合データセットでファインチューニング（Table VII の FT-LoRA 戦略に対応）
- 結合学習: M4Singer + Opencpop + MIR-ST500 の全データで学習（最良性能）

**データフォーマット**
- 音声 WAV + ノートレベル注釈 JSON
- フレームレベルの F0 軌跡
- ノート境界タイムスタンプ

---

### 1.4 各データセットの共通前処理パイプライン

```
raw_audio (44.1 kHz WAV)
    │
    ├─ resample → 24 kHz
    │
    ├─ EnCodec encoder → acoustic tokens a ∈ {0,...,N_q}^{8×T_a}
    │
    ├─ wav2vec/Whisper encoder → semantic tokens s ∈ ℝ^{D×T_s}
    │   └─ vector quantization → discrete semantic tokens n ∈ {0,...,V_s}^{T_s}
    │
    ├─ F0 extractor (CREPE or DIO) → continuous F0 → pitch quantizer
    │   └─ discrete pitch tokens m^d ∈ {0,...,C}^{T_p}
    │
    └─ lyrics (text) → pinyin conversion → phoneme sequence
```

セマンティックトークンの次元 D は MaskGCT の語彙設定に依存。ピッチ語彙サイズ C は論文で 128 または 256 程度が想定される（詳細はコードベース確認が必要）。

---

## 2. ソフトウェア依存関係

### 2.1 Python / PyTorch

| パッケージ | バージョン（推定） | 用途 |
|-----------|------------------|------|
| Python | 3.10 以上 | ランタイム |
| PyTorch | 2.1 以上（CUDA 12.x 対応） | 深層学習フレームワーク |
| torchaudio | 2.1 以上 | 音声処理 |
| numpy | 1.24 以上 | 数値計算 |
| librosa | 0.10 以上 | 音声特徴抽出・前処理 |
| soundfile | 0.12 以上 | WAV 読み書き |

論文本文に明示的なバージョン指定はないが、DiffLlama / MaskGCT の実装（2024–2025 年）に合わせると PyTorch 2.x が必要。

---

### 2.2 EnCodec（Meta's Audio Codec）

| 項目 | 内容 |
|------|------|
| 公式リポジトリ | https://github.com/facebookresearch/encodec |
| インストール | `pip install encodec` |
| 使用モデル | `encodec_24khz`（24 kHz 帯域、8 RVQ コードブック） |
| ライセンス | CC BY-NC 4.0 |

**論文での使用方法**
- 音声波形 → 8 層の RVQ アコースティックトークン列 `a` に圧縮
- S2A モデルが予測したトークンを EnCodec デコーダー `D_A` で波形に変換
- コードブック数: 8（論文 Fig.3 の `a ∈ {0,...,N_q}^8` に対応）
- 入力音声は 24 kHz にリサンプリング必須

**設定例**
```python
from encodec import EncodecModel
model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # 8 codebooks @ 75 Hz frame rate
```

---

### 2.3 MaskGCT フレームワーク（Amphion）

| 項目 | 内容 |
|------|------|
| リポジトリ | https://github.com/open-mmlab/Amphion |
| 対象モジュール | `models/tts/maskgct/` |
| ライセンス | MIT License |
| 論文参照 | Wang et al., 2024（MaskGCT: Zero-Shot TTS） |

**CoMelSinger での使用箇所**
1. **T2S モジュール（Text-to-Semantic）**: MaskGCT の T2S アーキテクチャをほぼそのまま流用。歌詞（テキスト）からセマンティックトークンを生成する非自己回帰マスク生成モデル。
2. **S2A モジュール（Semantic-to-Acoustic）**: MaskGCT の S2A アーキテクチャをベースに、LoRA でファインチューニング。DiffLlama スタイルの Transformer を使用。
3. **事前学習済みモデルの読み込み**: MaskGCT の公開チェックポイント（HuggingFace Hub: `amphion/MaskGCT`）を初期重みとして使用。

**インストール**
```bash
git clone https://github.com/open-mmlab/Amphion.git
cd Amphion
pip install -e ".[tts]"
```

または個別のパッケージとして：
```bash
pip install amphion
```

**MaskGCT 事前学習済みモデル（HuggingFace）**
- `amphion/MaskGCT-T2S`: テキスト→セマンティックトークン
- `amphion/MaskGCT-S2A`: セマンティック→アコースティックトークン
- これらを CoMelSinger の学習の初期点として使用

---

### 2.4 wav2vec / Whisper（セマンティック特徴抽出）

| 項目 | 内容 |
|------|------|
| 使用モデル（推定） | Whisper のエンコーダー、または wav2vec 2.0 Large |
| ライブラリ | `transformers`（HuggingFace）または `s3prl` |
| 主な用途 | 音声からセマンティックトークンを抽出（MaskGCT のパイプラインに準拠） |

**論文での言及**
- CLAUDE.md に「Semantic features: wav2vec-based (from Whisper/S3PRL)」と記載
- MaskGCT は Whisper のエンコーダーを semantic feature extractor として使用
- 抽出された連続特徴をベクトル量子化（VQ）して離散セマンティックトークンに変換

**インストール**
```bash
pip install transformers[torch]
pip install s3prl  # オプション（wav2vec 系モデルのアクセスに使用）
```

**コード例（Whisper ベース）**
```python
from transformers import WhisperModel, WhisperFeatureExtractor
# セマンティックエンコーダーとして Whisper エンコーダーを使用
model = WhisperModel.from_pretrained("openai/whisper-large-v3")
encoder = model.encoder
```

---

### 2.5 LoRA（PEFT ライブラリ）

| 項目 | 内容 |
|------|------|
| ライブラリ | `peft`（HuggingFace PEFT） |
| インストール | `pip install peft` |
| ライセンス | Apache 2.0 |
| 論文参照 | Hu et al. 2022、"Low-rank adaptation of large language models" |

**CoMelSinger での使用方法**
- S2A モデル（DiffLlama ベースの大規模モデル）を MIR-ST500 等の歌唱データで効率的にファインチューニング
- LoRA は DiffLlama の Linear 層に低ランク行列 A・B を挿入し、元の重みを凍結
- パラメータ効率的な学習により、GPU メモリを節約しながら歌唱ドメインに適応
- 論文 Table VII の比較実験（FT-LoRA vs FT-Pitch vs FT-All など）に対応

**設定例**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,              # ランク（論文には明記なし、推定）
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(s2a_model, lora_config)
```

---

### 2.6 その他の依存ライブラリ

| ライブラリ | 用途 |
|-----------|------|
| `crepe` または `pysptk` | F0（基本周波数）抽出 |
| `pypinyin` | 中国語テキスト → ピンイン変換 |
| `pyworld` | F0 抽出（DIO / Harvest アルゴリズム） |
| `mir_eval` | MIR-ST500 ピッチ評価指標の計算 |
| `sklearn` | 評価・前処理の補助 |
| `tensorboard` または `wandb` | 学習ログの可視化 |
| `hydra-core` | 設定管理（Amphion フレームワークで使用） |
| `accelerate` | 分散学習サポート |
| `einops` | Transformer の次元操作 |
| `flash-attn` | FlashAttention 2（高速 Attention、オプション） |

---

## 3. ハードウェア要件

### 3.1 GPU 構成（論文 Section IV.B より）

| モジュール | GPU | 台数 | 精度 |
|-----------|-----|------|------|
| SVT 学習 | NVIDIA RTX A5000 | 1 台 | FP32（推定） |
| S2A ファインチューニング | NVIDIA RTX A5000 | 4 台 | BF16 / FP16（推定） |

**NVIDIA RTX A5000 のスペック**
- VRAM: 24 GB GDDR6
- FP32 性能: 27.8 TFLOPS
- Tensor Core: あり（第 3 世代）
- NVLink: なし（PCIe 接続での 4-GPU 構成）

**学習の詳細（論文 Section IV.B より）**
- SVT:
  - オプティマイザ: AdamW
  - 学習率: 1e-4（コサインスケジューラー）
  - バッチサイズ: 32
  - エポック数: 100
- S2A ファインチューニング:
  - AdamW オプティマイザ
  - 学習率: 5e-5（コサインスケジューラー）
  - LoRA 適用で学習パラメータを削減
  - RTX A5000 × 4 GPU での分散学習
  - 対照学習バッチ: K = バッチ内のサンプル数（コサイン類似度行列の構成に使用）

### 3.2 メモリ・ストレージ要件（推定）

| 項目 | 推定容量 |
|------|---------|
| M4Singer 音声データ | 約 50–80 GB（元データ） |
| Opencpop 音声データ | 約 10–15 GB（元データ） |
| MIR-ST500 音声データ | 約 60–100 GB（元データ） |
| 前処理済みトークンキャッシュ | 約 50–100 GB |
| MaskGCT 事前学習モデル | 約 5–15 GB |
| EnCodec モデル | 約 500 MB |
| 中間チェックポイント | 約 10–30 GB |
| **合計ストレージ** | **約 300–400 GB** |

| 項目 | 推定容量 |
|------|---------|
| SVT 学習時 GPU VRAM | 24 GB × 1（RTX A5000 1 台で完結） |
| S2A 学習時 GPU VRAM | 24 GB × 4（RTX A5000 4 台） |
| ホスト RAM | 64 GB 以上推奨 |

---

## 4. 実装の主要コンポーネント

### 4.1 T2S モデル（MaskGCT ベース）

**役割**: 歌詞テキスト（ピンイン音素）→ 離散セマンティックトークン列

**アーキテクチャ**
- 非自己回帰マスク生成モデル（MaskGCT と同じパラダイム）
- 入力: ピンイン音素シーケンス + アコースティックプロンプト（参照音声）
- 出力: セマンティックトークン列 `n ∈ {0,...,V_s}^{T_s}`
- MaskGCT の事前学習済み T2S をそのまま転用（歌唱特化のファインチューニングなし、または最小限）

**実装ポイント**
- MaskGCT のマスク生成戦略を継承：推論時に繰り返しデノイジングで token を徐々に埋める
- 長さ予測器: テキスト長からセマンティックトークン長を予測
- アコースティックプロンプトはエンコードして条件入力に結合

---

### 4.2 S2A モデル（DiffLlama + MaskGCT）

**役割**: セマンティックトークン + 歌詞 + ピッチトークン + アコースティックプロンプト → アコースティックトークン

**アーキテクチャ（論文 Fig.3 参照）**

```
入力:
  e = f_S2A(B)           # セマンティックトークン埋め込み
  e' = f_S2A(B')         # セマンティックプロンプト埋め込み
  e_k を k∈[1:K_s] と K+1:K の領域に分割
  f^r = AvgPool(g^r)     # アコースティックプロンプトの平均
  g^r = AvgPool(g^r)     # 同上（ピッチ）

コンポーネント:
  - ピッチ埋め込み (e_p): ピッチトークン → 埋め込みベクトル、element-wise で e に加算
  - DiffLlama ベースの Transformer 本体
  - MaskLoss: ランダムマスクされたトークンの再構成損失
  - 長さ拡張 (Length Expansion): セマンティックトークン長をアコースティックトークン長に合わせる
```

**学習目標（3 つの損失の組み合わせ）**

```
L = λ_CL * L_CL + λ_SVT * L_SVT + λ_mask * L_mask   ... (9)
```

1. `L_mask`: マスク予測損失（MaskGCT 標準の損失）
2. `L_CL`: 対照学習損失（粗-細の 2 段階）
3. `L_SVT`: SVT 損失（ピッチ一致性）

**LoRA 適用箇所**
- DiffLlama の各 Transformer 層の Query・Value 行列
- 元の MaskGCT S2A 重みを凍結し、LoRA 差分のみ学習

---

### 4.3 SVT モジュール（軽量 Transformer）

**役割**: アコースティックトークン → フレームレベルのピッチトークン列を予測（補助監視信号として機能）

**アーキテクチャ**
- 軽量エンコーダのみ Transformer（Encoder-only）
- 入力: アコースティックトークン `a ∈ ℝ^{D×T_a}` + その周辺コンテキスト
- 出力: 各フレームに対応するピッチトークン `m^p ∈ {0,...,C}^{T_a}`
- ウィンドウサイズ: 12（ヘッド数 8、隠れ次元 512 → CLAUDE.md より）
- 各ピッチトークンは 12 個の離散クラスに分類（1 オクターブ 12 半音）

**損失関数（式 9 の L_SVT 部分）**

```
L_SVT = L_CE + λ_seg * L_seg + λ_dur * L_dur   ... (9)
```

- `L_CE`: クロスエントロピー損失（ピッチトークン分類）
- `L_seg`: セグメント遷移損失（ピッチ境界での差異ペナルティ）

```
L_seg = Σ [(1 - b_t) * ||p_t - p_{t-1}||^2 + b_t * max(0, δ - ||p_t - p_{t-1}||^2)]
```

ここで b_t はピッチ境界の二値マーカー、δ はマージン。

- `L_dur`: ソフトデュレーション損失（ピッチトークンの時間分布の滑らかさを促進）

```
L_dur = Σ_i (Σ_{t=T_i}^{T_i + a_i^d - 1} p_t[m_i^d] - a_i^d)^2
```

**学習プロセス**
1. SVT を M4Singer + Opencpop（+ MIR-ST500 でファインチューニング）で独立学習
2. 学習完了後、SVT の重みを凍結（`StopGrad` 操作）
3. S2A 学習中は SVT を固定補助監視として使用

---

### 4.4 対照学習損失関数

**2 段階の対照学習アーキテクチャ（論文 Fig.4 参照）**

#### 4.4.1 シーケンスレベル対照学習（SCL）

**目的**: グローバルな音響表現（音色）とピッチを分離し、プロンプトからの音色情報のみを抽出

```
S = [s_1, s_2, ..., s_K] # バッチ内のサンプル（同一歌手の異なるメロディー）

L_SCL = -1/(2K) * Σ log [exp(g^T g'^+ / τ) / Σ_{t≠j} exp(g^T g'^t / τ)]
                                                                        ... (3)
```

ここで:
- `g = AvgPool(g^r)`: アコースティックプロンプトのシーケンスレベル埋め込み
- `g^+`: 同一歌手の別メロディープロンプト（正例）
- `g^t (t≠j)`: 異なる歌手のプロンプト（負例）
- `τ`: 温度パラメータ

**バッチ構成**
- K 個のサンプル（S_A）: 通常サンプル（ピッチ条件なし）
- K 個のサンプル（S_A^f）: ピッチ条件あり
- S_B^f と S_B^a を組み合わせた混合バッチ B で学習

#### 4.4.2 フレームレベル対照学習（FCL）

**目的**: ローカルなフレームレベルでの音色とピッチの分離、ピッチの細粒度な一致性

```
L_FCL = -1/(L*L) * Σ_{i,j} Y^f[i,j] * log [exp(f^r[i]^T f^g[j] / τ) / Σ_k exp(f^r[i]^T f^g[k] / τ)]
                                                                        ... (5)
```

ここで:
- `f^r, f^g ∈ R^{L×D}`: フレームレベルのアコースティック埋め込み
- `Y^f[i,j]`: ソフトラベル行列（ピッチが一致するフレーム間で高い値）
- ソフトラベル = 同じピッチトークンを持つフレーム間で正の関係を割り当て

**ソフトラベルの利点**
- ピッチのパディング・繰り返しに対してロバスト
- 硬ラベル（完全一致）より自然なピッチ類似度を表現

**対角成分の対称化**
```
L_SCL_sym = (L_SCL(g^r, g^g) + L_SCL(g^g, g^r)) / 2   ... 定義 (3) の対称版
```

---

### 4.5 ピッチトークナイザー

**役割**: 連続 F0 値 → 離散ピッチトークンへの量子化

**設計**
- 入力: ground-truth F0（Hz 単位、0 は無声）
- 量子化: 音楽的なピッチスケール（等分平均律）に基づく
- 語彙サイズ C: 12 × オクターブ数 + 無声クラス（詳細はコード確認が必要）
- フレームレート: EnCodec と同じフレームレート（75 Hz @ 24 kHz）に合わせる

**処理フロー**
```
F0 (Hz) → MIDI ノート番号 → 12 クラス（半音）ラベル
         → one-hot → ピッチトークン m^d ∈ {0,...,C}
```

**ピッチシーケンスの正規化**
- 音符の持続時間でピッチトークンを繰り返し、フレームレベルアライメントを生成
- 無声フレームは特別なトークン（クラス 0 または C）で表現

---

### 4.6 EnCodec デコーダー

**役割**: S2A モデルが予測したアコースティックトークン → 音声波形

**設計**
- EnCodec デコーダー `D_A`（論文 Section III.A）
- 入力: 8 層の RVQ トークン列
- 出力: 24 kHz の音声波形
- 推論時は frozen（学習対象外）

**実装**
```python
from encodec import EncodecModel

encodec = EncodecModel.encodec_model_24khz()
encodec.eval()

# S2A が予測したトークン codes: (batch, 8, T_a)
with torch.no_grad():
    waveform = encodec.decode([(codes, None)])
```

---

## 5. 既存のオープンソースリソース

### 5.1 Amphion フレームワーク

| 項目 | 内容 |
|------|------|
| リポジトリ | https://github.com/open-mmlab/Amphion |
| ライセンス | MIT License |
| 管理組織 | OpenMMLab（Open-MMLab, Inc.） |
| 主な機能 | TTS / SVS / VC の統合フレームワーク、MaskGCT 実装を含む |

**CoMelSinger に関連する Amphion モジュール**
- `models/tts/maskgct/maskgct_t2s.py`: T2S モデル
- `models/tts/maskgct/maskgct_s2a.py`: S2A モデル
- `models/tts/maskgct/maskgct_inference.py`: 推論パイプライン
- `preprocessors/`: データ前処理スクリプト
- `bins/tts/train.py`: 学習エントリーポイント

**インストール手順**
```bash
git clone https://github.com/open-mmlab/Amphion.git
cd Amphion
conda create -n amphion python=3.10
conda activate amphion
pip install -e ".[tts]"
```

---

### 5.2 MaskGCT の公開実装

| 項目 | 内容 |
|------|------|
| 論文 | "MaskGCT: Zero-Shot TTS with Masked Generative Codec Transformer" (Wang et al., 2024) |
| HuggingFace | https://huggingface.co/amphion/MaskGCT |
| デモ | https://huggingface.co/spaces/amphion/MaskGCT |

**CoMelSinger が参照する MaskGCT の要素**
1. 非自己回帰マスク生成トレーニング戦略
2. セマンティックトークン / アコースティックトークンの 2 段階生成
3. アコースティックプロンプトによる話者条件付け
4. DiffLlama スタイルの Transformer（S2A 向け）

**MaskGCT 事前学習済みチェックポイントの読み込み**
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="amphion/MaskGCT", local_dir="./pretrained/maskgct")
```

---

### 5.3 利用可能な事前学習済みモデル

| モデル | 入手先 | 用途 |
|-------|--------|------|
| MaskGCT（T2S + S2A） | HuggingFace: `amphion/MaskGCT` | CoMelSinger の初期重み |
| EnCodec 24kHz | HuggingFace: `facebook/encodec_24khz` | アコースティックトークン化・デコード |
| Whisper Large v3 | HuggingFace: `openai/whisper-large-v3` | セマンティック特徴抽出 |
| wav2vec 2.0 Large | HuggingFace: `facebook/wav2vec2-large-960h` | セマンティック特徴抽出（代替） |

**CoMelSinger の公開リポジトリ（論文記載）**
- GitHub: https://github.com/ishine/CoMelSinger
- 2025 年 9 月時点では実装中の可能性あり（論文発表時点の状況による）

---

## 6. 環境構築手順（推奨）

```bash
# 1. リポジトリのクローン
git clone https://github.com/ishine/CoMelSinger.git
cd CoMelSinger

# 2. 仮想環境の作成
conda create -n comelsinger python=3.10
conda activate comelsinger

# 3. PyTorch のインストール（CUDA 12.x 対応）
pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# 4. Amphion フレームワーク
git clone https://github.com/open-mmlab/Amphion.git
cd Amphion && pip install -e ".[tts]" && cd ..

# 5. 主要依存ライブラリ
pip install encodec transformers peft accelerate
pip install librosa soundfile pyworld pypinyin
pip install tensorboard wandb mir_eval

# 6. 事前学習済みモデルのダウンロード
python -c "
from huggingface_hub import snapshot_download
snapshot_download('amphion/MaskGCT', local_dir='./pretrained/maskgct')
snapshot_download('facebook/encodec_24khz', local_dir='./pretrained/encodec')
"
```

---

## 7. 実装上の注意点

### 7.1 プロンプトからの音色漏れ（Prosody Leakage）の防止

論文の中心的な課題であり、以下の設計が連携して対処する：
- **対照学習（SCL + FCL）**: 音色とピッチを明示的に分離
- **SVT モジュール**: ピッチ監視信号を独立して提供
- **LoRA ファインチューニング**: 大規模事前学習の恩恵を保持しつつ歌唱ドメインに適応

### 7.2 フレームアライメント

ピッチトークン `m^d`、セマンティックトークン `s`、アコースティックトークン `a` のフレームレートが異なる場合がある。
- EnCodec 75 Hz（24 kHz, stride 320）
- Whisper 50 Hz（16 kHz, stride 320）
- F0 抽出レート: 通常 100 Hz または 200 Hz

フレームアライメントのため、ピッチシーケンスをアコースティックフレームに伸長・補間する処理が必要（論文式 (2) のデュレーション正規化に対応）。

### 7.3 評価スクリプト

論文で使用した評価指標の実装参照先：
- **MCD（Mel-Cepstral Distortion）**: `mir_eval` または独自実装
- **F0-RMSE**: `mir_eval.melody`
- **SingMOS**: https://github.com/South-Twilight/SingMOS
- **SECS（Speaker Embedding Cosine Similarity）**: WavLM-base+ 話者検証モデル

---

## 8. 参照論文・リソース一覧

| リソース | 参照 |
|---------|------|
| CoMelSinger 論文 | arXiv:2509.19883v2 |
| MaskGCT 論文 | Wang et al., 2024 (arXiv:2409.00750) |
| DiffLlama | Touvron et al., 2023 系（Meta LLaMA） |
| EnCodec | Défossez et al., 2022 |
| LoRA | Hu et al., 2022 |
| M4Singer データセット | Zhang et al., 2022（INTERSPEECH） |
| Opencpop データセット | Wang et al., 2022（INTERSPEECH） |
| MIR-ST500 データセット | Wang & Jang, 2021（ICASSP） |
| SingMOS 評価 | arXiv:2406.10911 |
| CoMelSinger GitHub | https://github.com/ishine/CoMelSinger |
| Amphion GitHub | https://github.com/open-mmlab/Amphion |
