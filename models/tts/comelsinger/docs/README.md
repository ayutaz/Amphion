# CoMelSinger 論文分析ドキュメント

> **対象論文**: "CoMelSinger: Discrete Token-Based Zero-Shot Singing Synthesis With Structured Melody Control and Guidance"
> **著者**: Jiunchuan Zhao, Wei Zeng, Tianle Lyu, Ye Wang (シンガポール国立大学)
> **arXiv**: 2509.19883v2 (2025年4月)

---

## ドキュメント一覧

### 基礎理解

| # | ファイル | 内容 | 行数 |
|---|---------|------|------|
| 01 | [01_overview.md](01_overview.md) | 論文概要・問題設定・主要貢献 | 272 |
| 02 | [02_architecture.md](02_architecture.md) | システムアーキテクチャ詳細（T2S + S2A） | 682 |

### コア技術

| # | ファイル | 内容 | 行数 |
|---|---------|------|------|
| 03 | [03_contrastive_learning.md](03_contrastive_learning.md) | Coarse-to-Fine 対照学習（SCL + FCL） | 265 |
| 04 | [04_svt_pitch_control.md](04_svt_pitch_control.md) | SVTモジュール・ピッチ制御メカニズム | 258 |
| 05 | [05_training_inference.md](05_training_inference.md) | 学習・推論パイプライン・損失関数 | 496 |

### 実験・評価

| # | ファイル | 内容 | 行数 |
|---|---------|------|------|
| 06 | [06_experiments.md](06_experiments.md) | 実験設定・結果・評価指標・Ablation Study | 347 |

### 関連技術調査

| # | ファイル | 内容 | 行数 |
|---|---------|------|------|
| 07 | [07_related_svs.md](07_related_svs.md) | SVS関連研究サーベイ（DiffSinger, MaskGCT等） | 760 |
| 08 | [08_discrete_token_codec.md](08_discrete_token_codec.md) | 離散トークン・ニューラルコーデック技術調査 | 1101 |

### 分析・実装

| # | ファイル | 内容 | 行数 |
|---|---------|------|------|
| 09 | [09_novelty_limitations.md](09_novelty_limitations.md) | 新規性分析・限界・今後の展望 | 504 |
| 10 | [10_implementation_requirements.md](10_implementation_requirements.md) | 再現実装に必要な要件・依存関係 | 647 |

**合計: 5,332行**

---

## CoMelSinger の核心（要約）

### 解決する問題
1. **Prosody Leakage（韻律漏れ）**: プロンプトベース合成で音響プロンプトのピッチ情報が生成音声に漏れる
2. **Melody Controllability（メロディ制御性）**: ゼロショット条件でフレームレベルの精密なピッチ制御が困難

### 提案手法
```
楽譜 (lyrics + pitch) ──→ [T2S] ──→ Semantic Tokens ──→ [S2A] ──→ Acoustic Tokens ──→ [EnCodec Decoder] ──→ 歌声
                                                           ↑
                                          Acoustic Prompt (音色参照)
                                          Contrastive Learning (韻律分離)
                                          SVT Module (ピッチ監督)
```

### 主要な技術的貢献
1. **MaskGCTベースの2段階SVSフレームワーク** — 離散トークンによる非自己回帰歌唱合成
2. **Coarse-to-Fine対照学習** — シーケンスレベル(SCL) + フレームレベル(FCL) でピッチと音色を分離
3. **SVTモジュール** — 軽量Transformerによるフレームレベルピッチ監督
4. **ゼロショット歌唱合成での最高性能** — MOS-Q 3.90, F0-RMSE 0.042, SECS 0.912

---

## 再現実装に向けて

再現実装の詳細は [10_implementation_requirements.md](10_implementation_requirements.md) を参照。主な依存関係：

- **ベースフレームワーク**: [Amphion/MaskGCT](https://github.com/open-mmlab/Amphion)
- **データセット**: M4Singer + Opencpop (+ MIR-ST500 for SVT)
- **ハードウェア**: NVIDIA RTX A5000 x4 (S2A学習)
- **主要ライブラリ**: PyTorch 2.1+, EnCodec, PEFT (LoRA), wav2vec/Whisper
