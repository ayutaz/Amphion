# CoMelSinger 関連研究：Singing Voice Synthesis (SVS) システム詳細調査

> 作成日: 2026-04-15  
> 対象論文: "CoMelSinger: Discrete Token-Based Zero-Shot Singing Synthesis With Structured Melody Control and Guidance" (arXiv: 2509.19883v2)

---

## 目次

1. [SVS技術の全体的進化](#1-svs技術の全体的進化)
2. [従来型SVSパイプライン](#2-従来型svs-パイプライン)
   - 2.1 HMMベースアプローチ
   - 2.2 VOCALOIDシステム
3. [End-to-End SVS](#3-end-to-end-svs)
   - 3.1 XiaoiceSing [38]
   - 3.2 DeepSinger [19]
4. [離散トークンベースSVS](#4-離散トークンベースsvs)
   - 4.1 DiffSinger [4]
   - 4.2 VISinger2 [9]
   - 4.3 StyleSinger [7]
   - 4.4 SPSinger [13]
   - 4.5 Vevo 1.5 [34]
5. [離散トークン生成パラダイム](#5-離散トークン生成パラダイム)
   - 5.1 Make-A-Voice [12]
6. [非自己回帰マスク生成モデル](#6-非自己回帰マスク生成モデル)
   - 6.1 MaskGCT [18]
7. [比較サマリー](#7-比較サマリー)

---

## 1. SVS技術の全体的進化

Singing Voice Synthesis (SVS) は、楽譜情報（歌詞・音符・デュレーション・ピッチ）から高品質な歌声を生成する技術である。この分野の発展は、大きく以下の4段階に分類できる。

```
第1世代: HMMベース（統計的パラメトリック合成）
    ↓
第2世代: DNN/End-to-End（ニューラルネットワーク導入）
    ↓
第3世代: 拡散モデル・GAN（生成品質の大幅向上）
    ↓
第4世代: 離散トークンベース（ゼロショット・未知話者対応）← 現在の最前線
```

CoMelSingerは第4世代に位置し、離散トークンとメロディ制御の組み合わせという新しいアプローチを提案している。

---

## 2. 従来型SVS パイプライン

### 2.1 HMMベースアプローチ（HMM-SVS）

#### 基本アーキテクチャ

HMM（隠れマルコフモデル）ベースのSVSは、テキスト音声合成（TTS）の統計的パラメトリック手法をSVSに応用したものである。主要なコンポーネントは以下の通り：

1. **言語解析フロントエンド**: 歌詞テキストを音韻情報（音素列）に変換
2. **HMMアコースティックモデル**: 音素ごとのスペクトル・F0（基本周波数）・デュレーションを統計的にモデル化
3. **パラメトリックボコーダ**: 統計モデルの出力パラメータから音声波形を再合成（STRAIGHT、WORLDなどが代表的）

典型的なモデルパラメータ:
- **スペクトル特徴**: メル周波数ケプストラム係数（MFCC）またはMGC（Mel-Generalized Cepstrum）
- **F0軌跡**: 対数スケールで表現、HMMの連続観測変数としてモデル化
- **非周期性指標（aperiodicity）**: 有声・無声の連続的な表現

代表的システムとして、2009年に提案されたEmons、2010年代のSinsy（日本語SVS）などがある。Sinsyは楽譜フォーマットとしてMusicXMLを採用し、HMMを用いて歌声パラメータを推定する。

#### 主要な特徴と限界

**強み:**
- 楽譜情報（音符境界・ピッチ）を明示的に入力として扱えるため、音符の忠実性が高い
- データ効率が比較的良好（数時間の録音データで動作可能）
- 制御性が高い（F0・デュレーションを直接操作できる）

**限界:**
- ボコーダによる音声品質の劣化（「機械音」感）
- 統計平均化による音声の平滑化（個性・表現力の欠如）
- 歌唱特有のビブラート・こぶしなどのモデル化が困難
- 話者依存性が高く、新しい歌手への適応にデータが必要

#### CoMelSingerとの関係

HMMベースSVSは制御性という観点でCoMelSingerと共通点を持つが、合成品質・ゼロショット能力の面で大きく異なる。CoMelSingerは離散トークンと大規模事前学習を活用することで、HMMが達成できなかった自然な歌唱表現と未知歌手への汎化を実現している。

---

### 2.2 VOCALOIDシステム

#### 基本アーキテクチャ

VOCALOIDはYamahaが開発した商業用歌唱合成システムで、2003年に初版がリリースされた。その技術的基盤は以下のコンポーネントから成る：

**VOCALOID 1・2（HMMベース + 単位選択）:**
- 歌手ごとの音声データベース（ダイフォン・トライフォンライブラリ）
- HMMによる変換モデル
- 単位選択（Unit Selection）アルゴリズム

**VOCALOID 3・4（統計的パラメトリック合成の強化）:**
- STRAIGHT/WORLDボコーダの採用
- Growth Parameter Control（GPC）による表現力強化
- クロス合成技術（複数歌手のブレンド）

**VOCALOID 5以降（ディープラーニングの導入）:**
- 深層ニューラルネットワークによるアコースティックモデルの更新
- より自然なビブラート・ブレスのモデル化

#### 主要な特徴と革新点

- **歌手キャラクター概念**: 初音ミクに代表されるバーチャルシンガーという文化的現象を創出
- **楽譜ベース制御**: ピアノロール形式の直感的なエディタ
- **精密な発音制御**: 日本語・英語・中国語等の多言語対応
- **ビブラート・ダイナミクス制御**: 細粒度の表現パラメータ

#### CoMelSingerとの関係

COmulSingerはVOCALOIDと同様に楽譜情報（ピッチ・デュレーション）を入力として使用するが、根本的なアーキテクチャが異なる。VOCALOIDが特定歌手のデータベースに依存するのに対し、CoMelSingerはゼロショット合成（未見歌手への汎化）を実現している点が決定的な違いである。また論文中でCoMelSingerは既存SVSシステムへの比較文脈として「VOCALOID系」が参照されており（論文p.1、イントロダクション部の記述）、CoMelSingerはこれらの既製システムとは異なる「フレキシブルな歌唱合成」を目指していることが示されている。

---

## 3. End-to-End SVS

### 3.1 XiaoiceSing [38]

> 論文内参照番号: [38]  
> 開発: Microsoft Research Asia  
> 発表: 2021年頃  
> フルタイトル: "XiaoiceSing: A Large-Scale High-Quality Chinese Singing Corpus and Singing Voice Synthesis System"

#### 基本アーキテクチャ

XiaoiceSingはMicrosoftが開発した大規模中国語歌唱合成システムで、以下のコンポーネントから構成される：

```
楽譜入力（音符・歌詞・デュレーション）
    ↓
FastSpeechベースのアコースティックモデル
    - 並列Transformer構造
    - デュレーション予測器（ソフトデュレーション損失）
    - F0予測モジュール
    ↓
HiFi-GANボコーダ（ニューラルボコーダ）
    ↓
歌声波形出力
```

**主要技術要素:**

1. **ソフトデュレーション損失 (Soft Duration Loss)**: 音符境界での急激な遷移を緩和し、滑らかなデュレーション予測を実現
2. **音素レベル・音符レベルのデュアル表現**: 歌唱特有の長い母音保持を表現するために、音節と音符の対応をモデル化
3. **F0予測の精緻化**: 歌唱における正確なピッチ追跡のため、F0予測モジュールを専用化
4. **大規模中国語コーパス**: 高品質な中国語歌唱データセット（XiaoiceSing Corpus）の構築と公開

#### 主要な特徴と革新点

- **FastSpeech的並列生成**: 自己回帰モデルより大幅に高速な推論
- **explicit duration modeling**: 楽譜のデュレーション情報を明示的にモデルに組み込み
- **end-to-end学習**: アコースティック特徴からニューラルボコーダまで一貫した学習
- **大規模コーパス整備**: 当時最大規模の中国語歌唱データの公開により、後続研究の基盤を提供

#### CoMelSingerとの関係

論文中でXiaoiceSingは主に2つの観点で参照されている：

1. **ソフトデュレーション損失の先行研究として**: CoMelSingerのSoft Duration Loss（論文式(8)）はXiaoiceSingのアプローチを参考にしており、論文p.6では「Inspired by recent advances in speech synthesis, we introduce a soft duration loss L_dur」と言及されている
2. **ベースラインシステムとして**: 歌唱合成の品質比較における参照点

**主な違い:**
- XiaoiceSingは学習済み歌手専用（ゼロショット不可）だが、CoMelSingerはゼロショット合成を実現
- XiaoiceSingは連続的なメルスペクトログラム予測ベースだが、CoMelSingerは離散トークンを使用
- CoMelSingerはメロディ制御と音色分離に特化した設計を持つ

---

### 3.2 DeepSinger [19]

> 論文内参照番号: [19]  
> 開発: Microsoft Research Asia（Baidu説もあり、論文表記ではBaidu関連研究として言及）  
> 発表: 2020年  
> フルタイトル: "DeepSinger: Singing Voice Synthesis with Data Mined from the Web"

#### 基本アーキテクチャ

DeepSingerはWebからの自動データ収集と多言語歌唱合成を特徴とするend-to-endシステムである：

```
Webクロール歌唱データ
    ↓
自動データパイプライン
    - 音源分離（Music Source Separation）
    - 自動歌詞アライメント
    - 品質フィルタリング
    ↓
Transformerベースアコースティックモデル
    - 歌詞・音符エンコーダ
    - Decoder（自己回帰）
    ↓
ボコーダ（World / HiFi-GAN）
    ↓
歌声出力
```

**主要技術要素:**

1. **Webマイニングパイプライン**: インターネット上の歌唱音声を自動収集・整形
2. **自動歌詞アライメント**: 音源分離後の歌声と歌詞の自動時間アライメント
3. **多言語対応**: 中国語・英語など複数言語の歌唱合成
4. **少データ学習**: Webデータを活用した大規模学習によりデータ収集コストを削減

#### 主要な特徴と革新点

- **データ収集の自動化**: 手動録音なしにWebからの自動データ収集を実現
- **多言語・多話者モデル**: 複数言語・歌手を単一モデルで処理
- **実用的なデータパイプライン**: 研究コミュニティへのデータ整備手法の提案

#### CoMelSingerとの関係

論文中でDeepSingerは以下の文脈で言及されている：

1. **プロミシティリーケージの先行研究として**: CoMelSingerの論文はDeepSingerを「音声プロミシティ（韻律・音色）が合成結果に不適切に漏れ込む問題」の代表的なケースとして参照。DeepSingerのような大規模TTSシステムは高い音声類似性を達成しているが、これは音声プロミシティからの意図しないリーケージに起因する可能性があると指摘されている（論文p.2）
2. **ベースラインとの性能比較**: Table IVの客観評価においてDeepSingerとの比較が記載

**主な違い:**
- DeepSingerはWebデータ収集・多言語対応に特化しているが、ゼロショット能力は限定的
- CoMelSingerはメロディ制御を明示的に設計し、音色とメロディの分離を重視

---

## 4. 離散トークンベースSVS

### 4.1 DiffSinger [4]

> 論文内参照番号: [4]  
> 発表: AAAI 2022  
> フルタイトル: "DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism"  
> 著者: Jinglin Liu et al.

#### 基本アーキテクチャ

DiffSingerは拡散確率モデル（Diffusion Probabilistic Models, DDPM）をSVSに応用した先駆的なシステムである：

```
入力: 歌詞 + 楽譜（音符列・デュレーション・ピッチ）
    ↓
テキストエンコーダ（Transformer）
    ↓
シャロー拡散メカニズム (Shallow Diffusion Mechanism)
    - 簡易なメルスペクトログラム推定モジュール
    - シンプルな拡散ステップ（浅い拡散）
    ↓
デノイジングネットワーク（WaveNet系）
    ↓
メルスペクトログラム
    ↓
HiFi-GANボコーダ
    ↓
音声波形
```

**主要技術要素:**

1. **シャロー拡散メカニズム**: 完全な拡散ではなく、簡易な推定からわずかな拡散ステップで高品質なメルスペクトログラムを生成。推論速度と品質のバランスを改善
2. **境界予測器**: 音素境界・音符境界の予測を明示的にモデル化
3. **F0予測モジュール**: 連続F0予測器を組み込み、正確なピッチ追跡を実現
4. **FastSpeech2との融合**: 事前推定モジュールとしてFastSpeech2系のアーキテクチャを採用

#### 主要な特徴と革新点

- **拡散モデルの歌唱への応用**: 画像生成で成功した拡散モデルを音声合成に適用
- **表現力の向上**: 従来のGANやオートエンコーダより多様で自然な歌声を生成
- **シャロー拡散による高速化**: 完全な拡散モデルより推論が高速
- **オープンソース実装**: GitHubでの公開により後続研究の基盤を提供

#### CoMelSingerとの関係

CoMelSinger論文のTable II（Evaluation Results）において、DiffSingerはベースラインシステムの一つとして比較されている：

| 指標 | DiffSinger | CoMelSinger |
|------|-----------|-------------|
| MOS-Q | 3.59 | **3.90** |
| MOS-N | 3.86 | **4.01** |
| SMOS | 3.91 | **4.22** |
| MCD | 5.36 | **4.17** |
| F0-RMSE | 0.084 | **0.042** |
| SingMOS | 4.13 | 4.32 |

（Table II、論文p.8より抜粋、数値は論文記載値）

**主な違い:**
- DiffSingerは連続的なメルスペクトログラムを直接予測するが、CoMelSingerは離散アコースティックトークンを使用
- DiffSingerはゼロショット合成を想定していないが、CoMelSingerは明示的にゼロショット対応
- CoMelSingerはメロディ制御に特化した専用モジュールを持つ

---

### 4.2 VISinger2 [9]

> 論文内参照番号: [9]  
> 発表: 2023年  
> フルタイトル: "VISinger 2: High-Fidelity End-to-End Singing Voice Synthesis Enhanced by Digital Signal Processing"  
> ベース: VITS2（Variational Inference TTS）

#### 基本アーキテクチャ

VISinger2はVITS（Variational Inference with adversarial learning for end-to-end TTS）フレームワークを歌唱合成に拡張したシステムである：

```
入力: 歌詞 + 楽譜（音符・ピッチ・デュレーション）
    ↓
テキストエンコーダ（Transformer）+ 音符エンコーダ
    ↓
変分オートエンコーダ (VAE)
    - エンコーダ: メルスペクトログラム → 潜在変数 z
    - デコーダ: 潜在変数 z → 音声波形
    ↓
Normalizing Flow
    ↓
GAN訓練（識別器との敵対的学習）
    ↓
DSP（デジタル信号処理）サイン合成モジュール
    ↓
最終音声出力
```

**主要技術要素:**

1. **VITSフレームワーク拡張**: end-to-end VAE + GAN の歌唱特化版
2. **DSPサイン合成**: 数式的なサイン波合成をニューラルネットワークと統合し、高忠実度のF0再現を実現
3. **音符レベルのアライメント**: 音符境界に合わせた内部アライメント学習
4. **Monotonic Alignment Search (MAS)**: VITSから継承した自動アライメント学習

#### 主要な特徴と革新点

- **高忠実度end-to-end合成**: ボコーダを分離せず、完全にend-to-endで高品質音声を生成
- **DSPサイン合成の統合**: 物理的なサイン波生成をDNNに組み込み、より正確なF0再現
- **潜在空間の表現力**: VAEによる豊かな潜在表現で多様な歌唱スタイルを捕捉

#### CoMelSingerとの関係

論文Table IIでの比較：

| 指標 | VISinger2 | CoMelSinger |
|------|----------|-------------|
| MOS-Q | 3.59 | **3.90** |
| F0-RMSE | 0.063 | **0.042** |
| SingMOS | 4.15 | 4.32 |

**主な違い:**
- VISinger2は特定の学習済み歌手専用（ゼロショット非対応）
- CoMelSingerは離散トークンベースで、VISinger2の連続潜在表現と根本的に異なる
- ゼロショット設定ではVISinger2はゼロショット用に設計されていないため直接比較困難

---

### 4.3 StyleSinger [7]

> 論文内参照番号: [7]  
> 発表: AAAI 2024  
> フルタイトル: "StyleSinger: Style Transfer for Out-Of-Domain Singing Voice Synthesis"

#### 基本アーキテクチャ

StyleSingerはスタイル転送（Style Transfer）の概念を歌唱合成に適用したシステムである：

```
入力: 歌詞 + 楽譜 + スタイル参照音声（プロンプト）
    ↓
スタイルエンコーダ
    - グローバルスタイル表現の抽出
    - スタイルトークン学習（GST: Global Style Tokens）
    ↓
テキスト・音符エンコーダ
    ↓
スタイル条件付きデコーダ
    ↓
メルスペクトログラム
    ↓
ボコーダ
    ↓
出力音声
```

**主要技術要素:**

1. **グローバルスタイルトークン (GST)**: 参照音声からスタイル情報（歌唱スタイル・感情・ビブラートなど）を抽出する注意機構ベースのトークン
2. **ドメイン外スタイル転送**: 学習データに含まれないスタイルへの汎化を目的とした設計
3. **Residual Style Adaptor**: スタイル情報の残差的統合
4. **スペクトル正規化GAN**: 合成品質向上のための敵対的学習

#### 主要な特徴と革新点

- **明示的スタイル制御**: 参照音声からのスタイル転送という直感的なインタフェース
- **ドメイン外汎化**: 学習データにないスタイルでもある程度の品質を維持
- **解釈可能なスタイルトークン**: スタイルの主要成分を分解・可視化できる

#### CoMelSingerとの関係

StyleSingerはCoMelSingerに近い「参照音声ベースの合成」というコンセプトを共有しているが、以下の点で異なる：

| 観点 | StyleSinger | CoMelSinger |
|------|------------|-------------|
| スタイル制御 | グローバルスタイルトークン | 離散アコースティックトークン + プロンプト |
| メロディ制御 | 楽譜情報を直接使用 | SVTモジュールによる明示的ピッチ監視 |
| ゼロショット | 限定的 | 明示的に設計 |
| 音色・メロディ分離 | 暗示的 | 明示的（コンタクティブ学習） |

論文Table IIでの比較：

| 指標 | StyleSinger | CoMelSinger |
|------|------------|-------------|
| MOS-Q | 3.67 | **3.90** |
| SMOS | 3.92 | **4.22** |
| F0-RMSE | 0.112 | **0.042** |

**主な違い:**
CoMelSingerはStyleSingerのスタイル制御の概念を発展させつつ、離散トークンベースの生成とより精密なメロディ監視を統合している。特にF0-RMSE（ピッチ精度）でCoMelSingerが大幅に優れており、メロディ制御の設計の違いが明確に現れている。

---

### 4.4 SPSinger [13]

> 論文内参照番号: [13]  
> フルタイトル: "SPSinger: Singing Voice Synthesis Based on Semi-supervised Pitch-guided Style Preference"  
> 特徴: ピッチガイドベースのスタイル学習

#### 基本アーキテクチャ

SPSingerは半教師あり学習とピッチガイドによるスタイル学習を組み合わせたSVSシステムである：

```
入力: 歌詞 + 楽譜 + ピッチ情報
    ↓
ピッチガイドエンコーダ
    - ピッチ情報から粗粒度のスタイル表現を抽出
    ↓
スタイル選好学習（Semi-supervised）
    - ラベルなしデータの活用
    - スタイル選好スコアの学習
    ↓
アコースティックデコーダ
    ↓
出力
```

**主要技術要素:**

1. **ピッチベーススタイルガイダンス**: ピッチ軌跡をスタイル制御の主要シグナルとして活用
2. **半教師あり学習**: ラベル付きデータが少ない状況での学習効率化
3. **スタイル選好モデリング**: 主観的な歌唱スタイルの好みを学習

#### CoMelSingerとの関係

論文Table IIでの比較：

| 指標 | SPSinger | CoMelSinger |
|------|---------|-------------|
| MOS-Q | 3.81 | **3.90** |
| SMOS | 4.06 | **4.22** |
| F0-RMSE | 0.112 | **0.042** |
| SingMOS | 4.28 | 4.32 |

SPSingerはCoMelSingerの主要ベースラインの一つであり、Table IVのゼロショット評価でも比較対象となっている。CoMelSingerはSPSingerに対し、ゼロショット設定・音色一貫性・ピッチ精度の全面で優れた性能を示している。

---

### 4.5 Vevo 1.5 [34]

> 論文内参照番号: [34]  
> 発表: 2024-2025年頃  
> フルタイトル: "Vevo: Controllable Zero-Shot Voice Imitation with Self-Supervised Disentanglement"（Vevo 1.5はその発展版）

#### 基本アーキテクチャ

Vevoは音声変換（Voice Conversion）と歌唱合成を統合したゼロショットシステムの最新版である：

```
入力: 歌唱コンテンツ + ターゲット話者参照
    ↓
自己教師あり音声エンコーダ（HuBERT等ベース）
    - コンテンツ表現（timbre非依存）
    - スピーカー表現
    ↓
ディセンタングルメント（コンテンツ・音色分離）
    ↓
トークンベース生成モデル
    ↓
ニューラルボコーダ
    ↓
出力歌声
```

**主要技術要素:**

1. **自己教師あり音声ディセンタングルメント**: コンテンツと音色を分離した潜在表現学習
2. **ゼロショット音声模倣**: 未見の話者/歌手への汎化
3. **離散トークンベース生成**: 最新の音声LMアプローチの採用
4. **コントローラブル生成**: 音色・スタイルの明示的制御

#### CoMelSingerとの関係

Vevo 1.5はCoMelSingerと同様に「ゼロショット歌唱合成」を目標とする最先端システムであり、直接的な競合関係にある：

論文Table IIでの比較：

| 指標 | Vevo 1.5 | CoMelSinger |
|------|---------|-------------|
| MOS-Q | 3.85 | **3.90** |
| SMOS | 4.17 | **4.22** |
| F0-RMSE | 0.051 | **0.042** |
| SingMOS | **4.39** | 4.32 |
| SECS | 0.907 | 0.912 |

**主な違い:**
- Vevo 1.5はSingMOSで最高スコアを記録しているが、CoMelSingerはF0-RMSE（ピッチ精度）・SECS（話者類似性）で優れている
- CoMelSingerはメロディ制御の明示的設計でVevo 1.5より高い音符忠実性を実現
- Vevo 1.5はコンテンツ・音色の自己教師あり分離に焦点を当てているが、CoMelSingerはSVTモジュールによる外部ピッチ監視を追加している

---

## 5. 離散トークン生成パラダイム

### 5.1 Make-A-Voice [12]

> 論文内参照番号: [12]  
> 発表: 2023年  
> フルタイトル: "Make-A-Voice: Unified Voice Synthesis With Discrete Representation"

#### 基本アーキテクチャ

Make-A-Voiceは音声・歌唱・音声変換を統一的な離散表現フレームワークで実現するシステムである：

```
入力: テキスト/歌詞 + 参照音声
    ↓
[音声エンコーダ]
HuBERT/音声コード化器
    → 意味トークン（Semantic Tokens）: コンテンツ情報
    ↓
[言語モデル段階]
Transformer LM（GPT系）
    → 音響トークン（Acoustic Tokens）の自己回帰生成
    ↓
[デコーダ段階]
音響コード化器デコーダ（EnCodecベース）
    ↓
音声波形
```

**主要技術要素:**

1. **離散意味トークン**: HuBERTなどの自己教師あり学習モデルから抽出した離散的なコンテンツ表現（k-meansクラスタリングによる量子化）
2. **階層的トークン表現**:
   - 第1層: 意味トークン（音素・内容情報）
   - 第2層: 音響トークン（音色・音質情報、EnCodec系）
3. **統一フレームワーク**: TTS・歌唱合成・Voice Conversionを単一のパラダイムで処理
4. **in-context学習**: 参照音声をプロンプトとして使用するゼロショット生成

#### 主要な特徴と革新点

- **離散トークンの統一パラダイム**: 音声の様々な側面を離散トークンで表現する汎用フレームワーク
- **LLMアーキテクチャの音声適用**: 大規模言語モデルの成功をそのまま音声生成に転用
- **マルチタスク対応**: 単一モデルで複数の音声合成タスクを処理
- **ゼロショット能力**: 参照音声のみで未知話者への合成が可能

#### CoMelSingerとの直接的な関係

Make-A-Voiceは**CoMelSingerの直接的な着想源の一つ**である。論文のイントロダクション（p.2）では：

> "Inspired by the success of large language models (LLMs), discrete token–based methods have achieved high performance in terms of speaker identity and style. This approach, exemplified by models such as VALL-E [15], encountered discrete representation high-fidelity speech synthesis by predicting multiple codes tokens per decoding step. Make-A-Voice [12] unifies speech and singing synthesis by predicting multiple codes tokens per decoding step."

と明記されており、Make-A-VoiceはCoMelSingerの「離散トークンによる統一的な歌唱合成」というコアコンセプトの直接的な先行研究として位置づけられている。

**CoMelSingerとの主な違い:**

| 観点 | Make-A-Voice | CoMelSinger |
|------|-------------|-------------|
| 生成方式 | 自己回帰（Autoregressive） | 非自己回帰（MaskGCTベース） |
| メロディ制御 | 楽譜情報の直接入力 | SVTモジュールによる明示的ピッチ監視 |
| 音色・メロディ分離 | 暗示的 | 明示的なコンタクティブ学習 |
| 推論速度 | 自己回帰のため低速 | 非自己回帰で高速 |

---

## 6. 非自己回帰マスク生成モデル

### 6.1 MaskGCT [18]

> 論文内参照番号: [18]  
> 発表: 2024年  
> フルタイトル: "MaskGCT: Zero-Shot TTS with Masked Generative Codec Transformer"  
> 開発: 中山大学・Microsoftなどの共同研究

#### 基本アーキテクチャ

MaskGCTは**CoMelSingerの直接的なベースシステム**であり、非自己回帰型のマスク生成トランスフォーマーを用いた高品質ゼロショットTTSシステムである：

```
入力: テキスト + 参照音声プロンプト
    ↓
[Semantic-to-Acoustic (S2A) 2段構成]
┌─────────────────────────────────────┐
│ Stage 1: テキスト → 意味トークン(T2S) │
│   - テキストエンコーダ               │
│   - 意味コード予測（w2v-BERT/HuBERT）│
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage 2: 意味 → 音響トークン (S2A)   │
│   - マスク生成Transformer            │
│   - MaskGITスタイルの並列デコード    │
│   - EnCodecベースの音響コード予測     │
└─────────────────────────────────────┘
    ↓
EnCodecデコーダ
    ↓
音声波形出力
```

**MaskGITスタイル生成の詳細:**

従来の自己回帰生成（左から右へ順番に生成）とは異なり、MaskGCTはマスク生成を使用する：

```
初期状態: [MASK][MASK][MASK]...[MASK]
    ↓ 反復1（全体の構造を予測）
[tok1][MASK][tok3]...[tokN]
    ↓ 反復2（信頼度の低いトークンを再マスク→再予測）
[tok1][tok2][tok3]...[tokN]
    ↓ ... (複数回反復)
最終出力: 完全なトークン列
```

**主要技術要素:**

1. **マスク生成トランスフォーマー (Masked Generative Transformer)**:
   - 全トークンを並列に予測（非自己回帰）
   - 信頼度スコアによる反復的な「マスク→予測」サイクル
   - 推論ステップ数をコントロール可能

2. **2段階の意味-音響分離**:
   - 意味トークン（S）: コンテンツ情報（音素・言語的内容）
   - 音響トークン（A）: 音色・音質情報（話者性・韻律）

3. **in-context学習によるゼロショット**:
   - 参照音声をプロンプト（prefix）として条件付け
   - 追加のfine-tuningなしで未見話者に対応

4. **条件付き確率モデル**:
   - p(a | s₁, s₂) のように意味トークン・音響トークン・外部条件を組み合わせた条件付き確率を学習

#### 主要な特徴と革新点

- **非自己回帰の高速推論**: 自己回帰TTSより大幅に高速な推論（並列生成）
- **in-context学習**: 明示的な話者適応学習なしでゼロショット合成
- **高品質ゼロショットTTS**: 最先端の自然性と話者類似性を達成
- **柔軟な生成長**: 非自己回帰設計により任意長の音声を生成可能

#### CoMelSingerとMaskGCTの関係（最重要）

MaskGCTはCoMelSingerの**直接的なベースモデル**である。CoMelSingerの論文では：

> "Building on this foundation, we introduce the MaskGCT work to singing synthesis. Our approach incorporates structured melody conditioning, and improved melody-timbre control in prompt-based synthesis to address the challenges of pitch-fidelity and melody controllability."

と明記されており、CoMelSingerはMaskGCTをSVSに拡張したシステムと理解できる。

**MaskGCTからCoMelSingerへの拡張点:**

```
MaskGCT (TTS用)
    ↓ 以下の拡張を追加
CoMelSinger (SVS用)

拡張1: 歌詞+楽譜入力
  - テキストプロンプト → 歌詞 + 音符シーケンス
  - デュレーション情報 + ピッチ情報の追加

拡張2: SVT (Singing Voice Transcription) モジュール
  - ピッチ監視のための補助SVTモジュール
  - 学習時: SVTからの明示的ピッチ監視信号
  - 推論時: SVTは不使用（ゼロショット対応）

拡張3: コンタクティブ学習
  - シーケンスレベル対照損失 (L_SCL)
  - フレームレベル対照損失 (L_FCL)
  - 音色とメロディの明示的分離

拡張4: ピッチ埋め込み統合
  - ピッチトークン m^p の導入
  - S2Aモデルへのピッチ埋め込みの追加 (e_p)

拡張5: ソフトデュレーション損失
  - 楽譜デュレーションへの軟化制約
  - リズム整合性の保証
```

**MaskGCTとCoMelSingerの技術的比較:**

| 要素 | MaskGCT (TTS) | CoMelSinger (SVS) |
|------|--------------|------------------|
| 入力 | テキスト + 参照音声 | 歌詞 + 楽譜 + 参照音声 |
| 生成ターゲット | 音響トークン | 音響トークン（ピッチ誘導付き） |
| ピッチ制御 | 自由（TTSは任意） | 明示的（楽譜のピッチに忠実） |
| ゼロショット | 話者のみ | 話者 + 歌手スタイル |
| 補助モジュール | なし | SVTモジュール、対照損失 |
| デュレーション制御 | フリー | ソフトデュレーション損失で拘束 |

---

## 7. 比較サマリー

### 各システムの技術的ポジショニング

```
           |  ゼロショット対応  |  離散トークン  |  明示的メロディ制御  |  高品質合成
-----------|-------------------|---------------|---------------------|----------
HMM-SVS    |       ✗           |      ✗        |       ○             |    ✗
VOCALOID   |       ✗           |      ✗        |       ○             |    △
XiaoiceSing|       ✗           |      ✗        |       ○             |    ○
DeepSinger |       △           |      ✗        |       ○             |    ○
DiffSinger |       ✗           |      ✗        |       ○             |    ○
VISinger2  |       ✗           |      ✗        |       ○             |    ○
StyleSinger|       △           |      ✗        |       ○             |    ○
SPSinger   |       △           |      ✗        |       ○             |    ○
Make-A-Voice|      ○           |      ○        |       △             |    ○
Vevo 1.5   |       ○           |      ○        |       △             |    ○
MaskGCT    |       ○(TTS)      |      ○        |       ✗(TTS不要)    |    ○
CoMelSinger|       ○           |      ○        |       ○(SVTモジュール)|   ○
```

### CoMelSingerが解決した主要課題

1. **プロミシティリーケージ問題（Prosody Leakage）**
   - 従来システム（特にTTSベースやMaskGCTそのまま）では、プロンプト音声の韻律・タイミング情報が生成物に不適切に漏れ込む
   - CoMelSingerは対照損失と明示的メロディ監視で音色とメロディを分離

2. **ゼロショット + 精密ピッチ制御の両立**
   - ゼロショット対応システム（Vevo 1.5、Make-A-Voice）は精密なピッチ制御が困難
   - 精密ピッチ制御システム（DiffSinger、VISinger2）はゼロショット非対応
   - CoMelSingerはSVTモジュールでこの両立を実現

3. **音符忠実性 vs. 自然性のトレードオフ**
   - F0-RMSE: CoMelSinger 0.042 vs. 競合システム 0.051〜0.112（全システム中最高精度）
   - MOS-Q: CoMelSinger 3.90（全システム中最高）

### 参考文献番号対照表

| 論文内番号 | システム名 | 対応する研究 |
|-----------|-----------|------------|
| [4] | DiffSinger | Liu et al., AAAI 2022 |
| [7] | StyleSinger | AAAI 2024 |
| [9] | VISinger2 | 2023年 |
| [12] | Make-A-Voice | 2023年 |
| [13] | SPSinger | ピッチガイドSVS |
| [18] | MaskGCT | 2024年、CoMelSingerの直接ベース |
| [19] | DeepSinger | Microsoft/Baidu 2020年 |
| [34] | Vevo 1.5 | ゼロショットSVS最新版 |
| [38] | XiaoiceSing | Microsoft 2021年 |

---

*本ドキュメントは CoMelSinger 論文 (arXiv: 2509.19883v2) の本文記述および著者の知識に基づき作成されました。Web検索による追加調査は現在の環境制約のため実施できなかった部分があります。各システムの詳細については、原著論文を参照してください。*
