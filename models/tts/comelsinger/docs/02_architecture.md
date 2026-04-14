# CoMelSinger システムアーキテクチャ詳細

> 論文: "CoMelSinger: Discrete Token-Based Zero-Shot Singing Synthesis With Structured Melody Control and Guidance"
> arXiv: 2509.19883v2 (2026年4月11日)
> 著者: Junchuan Zhao, Wei Zeng, Tianie Lyu, Ye Wang

---

## 1. 全体アーキテクチャ概要（2段階フレームワーク）

CoMelSinger は **2段階 (two-stage) のカスケード型フレームワーク** を採用している。

```
[入力]
  lyrics (歌詞テキスト)
  + pitch tokens (ピッチトークン: MIDIノートから変換)
  + acoustic prompt (音響プロンプト: 参照話者の短い音声)
        |
        v
┌─────────────────────────────────────┐
│  Stage 1: Text-to-Semantic (T2S)    │  ← MaskGCT ベースの非自己回帰型生成
│  歌詞+ピッチ → semantic tokens       │
└──────────────────┬──────────────────┘
                   │  semantic tokens (セマンティックトークン列)
                   v
┌─────────────────────────────────────┐
│  Stage 2: Semantic-to-Acoustic (S2A)│  ← DiffLlama バックボーン
│  semantic tokens → acoustic tokens  │     + マスクされた音響モデリング
└──────────────────┬──────────────────┘
                   │  acoustic tokens (RVQ 8層)
                   v
┌─────────────────────────────────────┐
│  Codec Decoder (コーデックデコーダ)  │  ← 事前学習済み音声コーデック
│  acoustic tokens → 波形 (waveform)  │
└─────────────────────────────────────┘
        |
        v
[出力] 合成された歌声の波形
```

### 設計思想

- **discrete tokens (離散トークン)** を中間表現として使用することで、zero-shot (ゼロショット: 未見の話者への対応) 汎化能力を実現
- **structured melody control (構造化されたメロディ制御)**: 歌声合成固有の課題である正確なピッチ制御を、トークンレベルで明示的に組み込む
- **prosody leakage (プロソディ漏洩) 問題の解決**: 音響プロンプトからピッチ情報が意図せず流入する問題に対し、コントラスティブ学習戦略で対処

---

## 2. Stage 1: Text-to-Semantic (T2S) — テキストからセマンティックへ

### 2.1 役割と概要

T2S モジュールは **歌詞テキスト・ピッチトークン・音響プロンプト** を受け取り、**semantic token (セマンティックトークン)** の列を生成する。生成された semantic tokens は話者の音色 (timbre) 情報を含まず、言語的・韻律的な内容のみをエンコードした中間表現である。

### 2.2 入力形式

| 入力 | 形式 | 説明 |
|---|---|---|
| lyrics (歌詞) | テキスト列 | 中国語/日本語等のフォネーム (phoneme) 列に変換後に使用 |
| pitch tokens (ピッチトークン) | 離散トークン列 | MIDIノートをトークン化したもの |
| acoustic prompt (音響プロンプト) | 短い音声クリップ | ターゲット話者の声質を示す参照音声 |

### 2.3 出力形式

- **semantic token sequence (セマンティックトークン列)**: `s = [s_1, ..., s_L]`
  - `L`: トークン列の長さ（フレーム数に対応）
  - 各トークンは離散インデックスで表現
  - 話者音色情報を含まない純粋な音声内容の表現

### 2.4 セマンティックトークンの抽出方法（wav2vec）

セマンティックトークンは事前学習済みの **wav2vec 2.0 (音声自己教師あり学習モデル)** から得られた連続表現を量子化して生成される。

```
音声波形
    |
    v
wav2vec 2.0 エンコーダ (音声の自己教師あり特徴抽出)
    |  連続特徴ベクトル
    v
Vector Quantization (ベクトル量子化)
    |  離散インデックス
    v
semantic tokens: s ∈ V_sem^L
  (V_sem: セマンティック語彙サイズ, L: 系列長)
```

- wav2vec から抽出した特徴は **prosodicな内容 (発話の韻律・言語内容) を反映** しつつ、話者固有の音色とは分離されやすい性質を持つ
- この性質が zero-shot 汎化に有利に働く

### 2.5 MaskGCT ベースの生成パラダイム

T2S は **MaskGCT (Masked Generative Codec Transformer: マスク生成コーデックトランスフォーマー)** のアーキテクチャを採用している。

#### MaskGCT の基本原理

MaskGCT は **非自己回帰型 (non-autoregressive: NAR)** の生成パラダイムであり、以下の特性を持つ。

1. **マスク予測学習 (masked prediction training)**:
   - 学習時: セマンティックトークン列の一部をランダムにマスク
   - モデルは文脈 (context) と外部条件付け入力 (conditioning inputs) からマスクされたトークンを予測
   - 条件付けは `P(m | s_unmasked, conditioning)` の形式

2. **並列デコーディング (parallel decoding)**:
   - 自己回帰型と異なり、複数のトークンを同時に生成可能
   - 推論速度が大幅に向上

3. **条件付けの形式**:

```
条件付き確率の定式化:
P(m | s_u, s_a, p_m, p̃_m)

  m   : マスクされた semantic tokens (予測対象)
  s_u : マスクされていない semantic tokens (文脈)
  s_a : 音響プロンプトから得た semantic tokens
  p_m : pitch-aligned (ピッチ対応) sequence
  p̃_m: prompt (プロンプト) の pitch tokens
```

#### Pitch-Aligned Sequence (ピッチ対応系列) の生成

ピッチトークンとセマンティックトークンの対応付けには **Length Expansion (長さ拡張)** モジュールが使用される。

```
pitch tokens m^d = [m_1^d, ..., m_S^d]
  (S: ピッチトークン数)
        |
        v
Length Expansion Module (長さ拡張モジュール)
  各ピッチトークンの duration (持続時間) を予測し
  フレームレベルに展開
        |
        v
pitch-aligned sequence m^a (フレーム対応ピッチ系列)
  長さ L のフレームレベルトークン列
```

**Duration (持続時間)** は以下の式で累積正規化される:

```
D = Σ_k m_k^d / Σ_j m_j^d

各フレームのスパンは cumulative normalized duration から算出
```

---

## 3. Stage 2: Semantic-to-Acoustic (S2A) — セマンティックから音響へ

### 3.1 役割と概要

S2A モジュールは T2S が生成した **semantic tokens** を受け取り、音色・音質情報を含む **acoustic tokens (音響トークン)** を予測する。この acoustic tokens が最終的に音声コーデックデコーダを通じて波形に変換される。

### 3.2 入力条件

S2A モジュールには以下の3種類の条件付け入力が与えられる:

#### (a) lyrics (歌詞) + pitch tokens (ピッチトークン)

```
入力形式:
  lyrics      → phoneme embeddings (フォネーム埋め込み)
  pitch tokens → pitch embeddings  (ピッチ埋め込み)
                           |
                           v
  要素ごとに合算 (element-wise addition):
  e_i = phoneme_emb_i + pitch_emb_i
  (i: フレームインデックス)
```

これにより、**メロディ情報を音節レベルで条件として付加**する。

#### (b) acoustic prompt (音響プロンプト: a')

参照話者の音声から抽出した音響プロンプトで、ターゲット話者の **timbre (音色)** をモデルに伝える。

```
参照音声 → 事前学習済みコーデック → acoustic tokens → a'
                                              ↓
                               S2A への条件付けとして連結
```

#### (c) semantic tokens (セマンティックトークン: e)

T2S が生成した semantic tokens で、言語・韻律内容を担う。

```
T2S 出力 s → semantic embedding → e ∈ R^{L×D}
```

**最終的な S2A への入力ベクトル**:

```
e_composite = e + e_pitch
  + (acoustic prompt の連結)

コンディショニング入力 = e_composite ∪ acoustic prompt
```

### 3.3 DiffLlama-Style バックボーン

S2A のバックボーンは **DiffLlama (拡散型 Llama: 大規模言語モデルを拡散モデルに適用)** スタイルのアーキテクチャを採用している。

#### 特徴

1. **Llama-style Transformer (Llama 型トランスフォーマー)**:
   - RoPE (Rotary Position Embedding: 回転位置エンコーディング) を使用
   - 大規模言語モデルの Llama アーキテクチャを基盤とする

2. **Low-Rank Adaptation (LoRA: 低ランク適応)**:
   - 事前学習済みモデルのパラメータを凍結
   - LoRA により低ランク行列だけを更新
   - 歌声合成ドメインへの効率的な適応を実現

3. **拡散 (diffusion) スタイルの学習**:
   - ノイズを付加した入力から元のトークンを復元するデノイジング (denoising) 学習
   - 確率的生成モデルの枠組みで動作

```
S2A Forward Pass (順伝播):

noisy acoustic tokens ã (ランダムマスク済み)
        +
semantic embedding e
        +
pitch embedding e_pitch
        +
acoustic prompt a'
        |
        v
DiffLlama Backbone
  [Transformer Layer 1]
  [Transformer Layer 2]
  ...
  [Transformer Layer N]
        |
        v
masked token prediction (マスクされたトークンの予測)
        |
        v
acoustic tokens a = [a^1, ..., a^8]  (8層 RVQ)
```

### 3.4 マスクされた音響モデリング (Masked Acoustic Modeling)

S2A の学習目標の中核は **mask prediction loss (マスク予測損失)** である。

#### 動作メカニズム

```
学習時:
1. 正解 acoustic tokens a (8層 RVQ) をランダムにマスク
2. マスクされたトークンを [MASK] トークンで置換
3. noisy な acoustic tokens ã を S2A の入力とする
4. S2A は条件付き入力 (e, a') を参照しながら
   マスクされた位置のトークンを予測

損失関数 L_mask:
  L_mask = MaskLoss(â, a)
  (â: 予測されたトークン, a: 正解トークン)
```

#### S2A 全体の学習目標

S2A の学習は以下の3つの損失の組み合わせで行われる:

```
L_total = λ_CL · L_CL + λ_SVT · L_SVT + λ_mask · L_mask

(1) L_mask: mask prediction loss (マスク予測損失)
    ランダムにマスクした音響トークンを
    デノイジングデコーダで再構成

(2) L_CL: coarse-to-fine contrastive loss
    (粗から細へのコントラスティブ損失)
    → 後述のコントラスティブ学習戦略を参照

(3) L_SVT: Singing Voice Transcription loss
    (歌声転写補助損失)
    SVT モジュールからのピッチ教師信号
```

### 3.5 RVQ Codebook 構造（8層）

CoMelSinger では **RVQ (Residual Vector Quantization: 残差ベクトル量子化)** に基づく音声コーデックを使用する。

#### RVQ の原理

```
音声波形
    |
    v
Codec Encoder (コーデックエンコーダ)
    |
    v
連続音声特徴 z ∈ R^{T×D}
    |
    v
RVQ 量子化 (8段階の残差量子化):

  Layer 1: z_1 = Quantize(z)         → code_1
  Layer 2: r_1 = z - z_1
           z_2 = Quantize(r_1)       → code_2
  Layer 3: r_2 = r_1 - z_2
           z_3 = Quantize(r_2)       → code_3
  ...
  Layer 8: z_8 = Quantize(r_7)       → code_8

  各層のコードブック (codebook) サイズ: 通常 1024
  24 kHz で均一にダウンサンプリング
    |
    v
acoustic tokens a = (code_1, ..., code_8) ∈ Z^{T×8}
```

#### RVQ の役割分担

| 層 | 役割 |
|---|---|
| Layer 1 (第1層) | 粗い音声構造 (主要なスペクトル形状) |
| Layer 2-4 (第2-4層) | 中間的な音声詳細 |
| Layer 5-8 (第5-8層) | 細かい音声ディテール (高周波成分) |

- S2A は **全8層の acoustic tokens を同時に予測** する
- 事前学習済み codec のデコーダが8層のトークンから波形を再構成

```
a = (code_1, ..., code_8)
        |
        v
Codec Decoder (コーデックデコーダ) D_A
        |
        v
波形 w = D_A(a)
```

---

## 4. 各モジュール間のデータフロー

### 4.1 詳細なデータフロー図

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[学習前準備]

参照音声
    └─→ wav2vec 2.0 → VQ → semantic tokens (セマンティックトークン)
    └─→ 音声コーデック → RVQ → acoustic tokens, a = V_acous^{L×8}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Stage 1: T2S — 学習・推論共通]

  Input (入力):
  ┌──────────────────────────────┐
  │ lyrics (歌詞)                 │──→ phoneme embedding
  │ pitch tokens m^d             │──→ Length Expansion → m^a (frame-level)
  │ acoustic prompt s^' (semantic)│──→ context embedding
  └──────────────────────────────┘
              |
              v
  MaskGCT Transformer
  (Non-Autoregressive Masked Generation)
              |
              v
  semantic tokens s ∈ V_sem^L   ← T2S の出力

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Stage 2: S2A — 学習・推論共通]

  Input (入力):
  ┌──────────────────────────────────┐
  │ e = f_S2A(s)  ← T2S の semantic  │
  │   + e_pitch   ← ピッチ埋め込み   │  → 複合条件ベクトル
  │ a' (acoustic prompt トークン)     │
  └──────────────────────────────────┘
              |
              v
  DiffLlama Backbone
  (マスク予測 + 拡散スタイル学習)
              |
              v
  acoustic tokens â = (code_1, ..., code_8) ∈ Z^{L×8}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[最終デコード]

  acoustic tokens â
              |
              v
  Codec Decoder D_A (事前学習済み、凍結)
              |
              v
  合成歌声波形 w ∈ R^{T_audio}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 4.2 補助モジュール: SVT (Singing Voice Transcription)

学習中のみ使用される **SVT モジュール** は、S2A に対してピッチ整合の監督信号を提供する。

```
[SVT モジュール (学習時のみ)]

acoustic tokens a
        |
        v
SVT Encoder (軽量エンコーダ)
  Transformer (encoder-only)
  入力: 12 個の離散コード × 512次元 × 8 attention heads
        |
        v
predicted pitch token sequence m^p ∈ R^{L×C}
  (L: フレーム数, C: ピッチ語彙サイズ)
        |
        v
L_SVT = L_CE + λ_seg · L_seg + λ_dur · L_dur

  L_CE  : cross-entropy loss (交差エントロピー損失)
  L_seg : segment transition loss (音節境界損失)
  L_dur : soft duration loss (ソフト持続時間損失)
```

**SVT モジュールは推論時には使用されず**、S2A の学習完了後に凍結される。

---

## 5. 推論時パイプライン（MaskGCT の反復デコーディング）

### 5.1 推論の全体フロー

```
[推論入力]
  lyrics: テキスト
  score:  楽譜 (MIDIノートとして表現されたピッチ・デュレーション)
  acoustic prompt: 短い参照音声 (数秒)

        |
        v

[前処理]
  lyrics  → G2P (Grapheme-to-Phoneme) 変換 → phoneme tokens
  score   → pitch tokens m^d (離散ピッチトークン)
  prompt  → wav2vec 2.0 → semantic tokens s^'
  prompt  → 音声コーデック → acoustic tokens a'

        |
        v

[Stage 1: T2S 反復デコーディング]
  MaskGCT 反復デコーディング (iterative decoding)
  ┌──────────────────────────────────────────┐
  │ 初期状態: 全トークンをマスク             │
  │                                          │
  │ Iteration 1: 高信頼度トークンのみ確定   │
  │   [MASK][MASK]...[MASK]                  │
  │        ↓ (信頼度上位 r_1% を確定)       │
  │   s_1  [MASK]  s_3  [MASK]  ...         │
  │                                          │
  │ Iteration 2: 残りのマスクから再予測      │
  │        ↓ (信頼度上位 r_2% を確定)       │
  │   s_1   s_2    s_3   s_4   ...          │
  │                                          │
  │ ...                                      │
  │                                          │
  │ Iteration T: 全トークン確定             │
  │   s_1   s_2    s_3   s_4   ...  s_L     │
  └──────────────────────────────────────────┘
        |
        v semantic tokens s (確定)

[Stage 2: S2A 反復デコーディング]
  同様の MaskGCT 反復デコーディング
  ┌──────────────────────────────────────────┐
  │ 初期状態: 全8層の acoustic tokens をマスク│
  │                                          │
  │ 各イテレーションで信頼度スコアに基づき   │
  │ トークンを段階的に確定                   │
  │                                          │
  │ 条件付き入力:                            │
  │   - semantic tokens e (T2Sの出力)        │
  │   - pitch-aligned embedding e_pitch      │
  │   - acoustic prompt a'                   │
  └──────────────────────────────────────────┘
        |
        v acoustic tokens â ∈ Z^{L×8} (確定)

[後処理]
  acoustic tokens → Codec Decoder D_A → 合成歌声波形

[出力]
  合成された歌声の wav ファイル
```

### 5.2 MaskGCT 反復デコーディングの詳細アルゴリズム

MaskGCT の推論は **confidence-based masking (信頼度ベースのマスキング)** 戦略により段階的にトークンを確定させる。

```
入力: 生成長 L, 反復回数 T, 条件付け情報 C
出力: 生成トークン列 s ∈ {1,...,V}^L

初期化: s^0 = [MASK, MASK, ..., MASK]  (全マスク)

for t = 1 to T:
  1. Forward pass (順伝播):
     logits = Model(s^{t-1}, C)
     probs  = softmax(logits)

  2. サンプリング:
     s^{t}_{candidate} = sample(probs)

  3. 信頼度スコア計算:
     confidence_i = max_v(probs_{i,v})  for i in [1..L]

  4. マスク率の計算:
     r_t = cos_schedule(t/T)  ← コサインスケジュールに従い減少
     n_mask = ⌊r_t × L⌋

  5. 下位信頼度のトークンを再マスク:
     masked_indices = argsort(confidence)[:n_mask]
     s^{t}[masked_indices] = [MASK]

return s^T  (全トークン確定)
```

### 5.3 推論時のピッチ同期メカニズム

推論時に **ピッチトークンとセマンティックトークンの時間的同期** を保つため、以下の処理を行う。

```
pitch token m^d (持続時間情報付き):
  [m_1, dur_1], [m_2, dur_2], ..., [m_S, dur_S]
          |
          v
Duration Normalization (持続時間正規化):
  a_i^d = ⌊(m_i^d × L) / Σ_j m_j^d⌋
          |
          v
frame-aligned pitch sequence m^a (フレーム対応ピッチ系列):
  長さ L のフレームレベルトークン列
  S2A の条件付け入力として使用
```

---

## 6. コントラスティブ学習戦略（Prosody Leakage 対策）

### 6.1 Sequence-Level Contrastive Learning (系列レベルコントラスティブ学習)

```
同一歌手の異なる発話サンプルを収集:

Batch B = {(x_1,...,x_K)}

S_B^A (音響トークンサンプル群) と
S_B^f (フレームレベル特徴群) を構成

音響プロンプト a, a^r ← PromptGen(B, B^r)
               ↓
e  = f_{S2A}(B )  (B のセマンティック埋め込み)
e' = f_{S2A}(B')  (B' のセマンティック埋め込み)

g  = AvgPool(e^r)   (平均プール → 系列レベル表現)
ĝ  = AvgPool(e'^r)

L_{SCL}: g と ĝ の対称コントラスティブ損失
  → 同一歌手の埋め込みを近づけ、
    異なる歌手の埋め込みを遠ざける
```

### 6.2 Frame-Level Contrastive Learning (フレームレベルコントラスティブ学習)

```
局所ピッチ変動に対する音響整合:

f  = e_{1:K_s}     (フレームレベル特徴の前半)
F  = e'_{K_s+1:K}  (フレームレベル特徴の後半)
     ↓
L_{FCL}: f と F の対称コントラスティブ損失
  → フレームレベルでのピッチ一貫性を確保
  → 異なるメロディ間での音色分離を実現
```

### 6.3 損失の組み合わせ

```
最終学習損失:
L = λ_{CL} · L_{CL} + λ_{SVT} · L_{SVT} + λ_{mask} · L_{mask}

  L_{CL}  = L_{SCL} + L_{FCL}   (系列 + フレームのコントラスティブ)
  L_{SVT} = L_{CE} + λ_{seg} · L_{seg} + λ_{dur} · L_{dur}  (SVT補助損失)
  L_{mask}: マスク予測損失 (主タスク)

デフォルト重み:
  λ_{CL}   = 1.0
  λ_{SVT}  = 1.0
  λ_{mask} = 1.0
```

---

## 7. 学習手順

### 7.1 2段階学習プロセス

```
Phase 1: SVT モジュールの事前学習
  ├─ 最適化対象: SVT モデルパラメータ
  ├─ 損失関数: L_{CE} (ピッチトークン予測)
  └─ 完了後: SVT パラメータを凍結

Phase 2: S2A モジュールのファインチューニング
  ├─ 最適化対象: S2A パラメータ θ
  ├─ 凍結: SVT モデル f_{SVT} (StopGrad を適用)
  ├─ LoRA: 事前学習済み DiffLlama の低ランク行列のみ更新
  └─ 損失関数: L = λ_{CL}·L_{CL} + λ_{SVT}·L_{SVT} + λ_{mask}·L_{mask}
```

**Algorithm 1: Finetuning S2A with Contrastive Learning and SVT Supervision** の疑似コード:

```
Require: S2A モデルパラメータ θ; 凍結 SVT モデル f_SVT;
         学習データ D; 損失重み λ_{CL}, λ_{SVT}, λ_{mask}; エポック数 N; 学習率 η

Ensure: 学習済みパラメータ θ

for i = 1 to N do:
  バッチ B = {x_1,...,x_K} をサンプル
  S_B^A と S_B^f を構成
  B' = P(S_B^A)  (プロンプト選択)
  a, a^r ← PromptGen(B), a'^r ← PromptGen(B')

  Forward pass:
    e ← f_{S2A}(B), e' ← f_{S2A}(B')
    (e_{1:K_s}, e_{K_s+1:K}) に分割
    g  ← AvgPool(e^r), ĝ ← AvgPool(e'^r)
    â ← f_{head}(ã)  (デノイジングデコーダ)

  損失計算:
    L_{SCL} を g, ĝ から計算
    L_{FCL} を f, F から計算
    L_{CL} を計算
    â ← f_{head}(ã); L_{mask} = MaskLoss(â, a)
    m^p ← f_{SVT}(StopGrad(â)); L_{SVT} を m^p, m^a から計算
    L ← λ_{CL}·L_{CL} + λ_{SVT}·L_{SVT} + λ_{mask}·L_{mask}

  パラメータ更新:
    θ ← θ - η · ∇_θ L
end for
```

### 7.2 実装詳細

| 設定項目 | 値 |
|---|---|
| GPU | NVIDIA RTX A5000 GPU (1枚) |
| オプティマイザ | AdamW |
| 学習率 | 5×10^{-5} |
| コサイン学習率スケジュール | 減衰率 0.01 |
| エポック数 | 100 |
| バッチサイズ | 32 |
| S2A ファインチューニング | NVIDIA RTX A5000 GPU (複数枚) |
| S2A 学習率 | 同様の AdamW 設定 |
| S2A LoRA ランク | 低ランク行列 (具体値は実装依存) |
| コントラスティブ学習サンプル数 | K 個 (50%がプロンプト用、残り50%が生成用) |

---

## 8. まとめ: アーキテクチャの特徴と革新点

| 特徴 | 説明 |
|---|---|
| **2段階カスケード** | T2S + S2A の分離により音色と内容を独立制御 |
| **離散トークン中間表現** | zero-shot 汎化を促進、エンドツーエンドより柔軟 |
| **MaskGCT 非自己回帰生成** | 並列デコードで推論速度向上、文脈整合性も維持 |
| **RVQ 8層コーデック** | 高品質な音声再合成を離散空間で実現 |
| **SVT 補助督** | ピッチ整合の明示的な監督信号でメロディ忠実度を向上 |
| **Coarse-to-Fine コントラスティブ学習** | prosody leakage を系列・フレームの両レベルで抑制 |
| **DiffLlama + LoRA** | 大規模事前学習モデルを効率的に歌声合成ドメインへ適応 |
