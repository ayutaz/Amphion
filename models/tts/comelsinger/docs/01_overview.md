# CoMelSinger 論文概要・問題設定・主要貢献

## 1. 論文の全体概要

### タイトル

**CoMelSinger: Discrete Token-Based Zero-Shot Singing Voice Synthesis with Structured Melody Control and Guidance**

### 著者・所属

- **Jianchuan Zhao（趙建川）**
- **Wei Zeng（曾偉）**
- **Tianie Lyu（呂天俄）**
- **Ye Wang（王野）** — IEEE Member

所属：シンガポール国立大学（National University of Singapore）

### 発表年・arXiv情報

- arXiv投稿日：2025年4月11日
- arXiv ID：arXiv:2509.19883v2
- 掲載想定：IEEE（査読付き論文誌・国際会議）

---

## 2. 解決しようとしている問題

### 2-1. Singing Voice Synthesis（SVS：歌声合成）の概要と近年の潮流

Singing Voice Synthesis（SVS、歌声合成）は、楽譜情報（音符のピッチと持続時間）および歌詞テキストから高品質な歌声を自動生成する技術である。近年、Discrete Token-Based（離散トークンベース）アーキテクチャが Text-to-Speech（TTS、テキスト音声合成）分野で成功を収めており、その技術をSVSへ応用する研究が活発になっている。

離散トークンベースのSVSでは、音声をQuantized（量子化された）離散コードとして表現し、大規模言語モデルの枠組みを活用して音声を生成する。この手法は、以下の点で優れている：

- **In-context learning（文脈内学習）**：参照音声（プロンプト）から話者の音色・スタイルを直接模倣できる
- **Zero-shot generalization（ゼロショット汎化）**：未見の話者の音声を生成できる
- **非自己回帰（Non-Autoregressive）生成**：高速かつ並列な推論が可能

代表的なモデルとして、MaskGCT（Masked Generative Codec Transformer）が挙げられ、これを基盤として本論文の手法が構築されている。

### 2-2. 主要課題①：Prosody Leakage（韻律漏れ）

Prosody Leakage（韻律漏れ）とは、**音声プロンプトに含まれる韻律情報（ピッチ・エネルギー・タイミング）が、生成された音声へ意図せず混入してしまう現象**である。

TTS分野では、プロンプトから話者の表現スタイルを模倣することが有益な機能とみなされている。しかし、SVSにおいては根本的な問題をはらんでいる。歌声合成では、ピッチとリズムは音楽スコア（楽譜）によって厳密に支配されるべきであり、音声プロンプトの韻律に引きずられることは合成品質を著しく損なう。

具体的には：

- 音声プロンプトの**音色（Timbre）と韻律（Prosody）が絡み合う**（Entanglement）
- プロンプトが同時に**音色・声質・メロディー・タイミング**に影響を与えてしまう
- 結果として、楽譜で指定したメロディーとプロンプトの韻律の間で精度が低下し、**ティンバー（音色）とメロディーを精密に分離できない**

この問題は特に、訓練データの多様性が限られた Discrete Token-Based アーキテクチャにおいて顕著である。歌唱データセットは一般的に話者数・多様性ともにTTSデータより少なく、in-context learning の学習が困難になる。

### 2-3. 主要課題②：Melody Controllability（メロディー制御性）の限界

既存のToken-Based SVSシステムは、メロディー制御に関して重大な限界を抱えている：

- **Coarse-to-Fine（粗から細への）制御が欠如**：音符単位の粗い制御しか持たず、フレームレベルの精細な制御ができない
- **ピッチトークンの表現力の限界**：音高は量子化されているため、細かいビブラート・グリッサンドなどの表現が損なわれる
- **ゼロショット設定での制御の難しさ**：未見の歌手（Unseen Singer）に対し、楽譜に忠実なメロディー生成と音色保持の両立が困難
- **TSSモデルからの直接転用の問題**：TTS向けのToken-BasedアーキテクチャをそのままSVSへ適用しても、歌声特有の細粒度（Fine-grained）ピッチ制御が実現できない

### 2-4. 問題の整理

| 課題 | 内容 | 影響 |
|------|------|------|
| Prosody Leakage（韻律漏れ） | プロンプトの韻律が生成音声に混入 | 楽譜のメロディーに沿えない |
| Melody Controllability（メロディー制御性） | フレームレベルの精細なピッチ制御が困難 | 表現豊かな歌声の生成が難しい |
| Timbre-Melody Entanglement（音色とメロディーの絡み合い） | 音色とピッチを分離できない | ゼロショット条件での品質低下 |
| Limited Training Data（学習データの限界） | 歌唱データは話者多様性が低い | 汎化性能の低下 |

---

## 3. 主要な貢献（4つの主要貢献）

### 貢献①：CoMelSinger フレームワークの提案

**CoMelSinger**は、Discrete Token-Based（離散トークンベース）アーキテクチャを採用しながら、**構造化されたメロディー制御とガイダンス**を統合した新しいSVSフレームワークである。

具体的な特徴：
- **MaskGCT（Masked Generative Codec Transformer）**を基盤とした非自己回帰型生成
- **Two-Stage Pipeline（二段階パイプライン）**：
  - Stage 1（T2S：Text-to-Semantic）：歌詞・音符・プロンプトから Semantic Token（意味トークン）を生成
  - Stage 2（S2A：Semantic-to-Acoustic）：Semantic Token から Acoustic Token（音響トークン）を生成し、最終的に波形を再構成
- **離散的な中間表現（Discrete Intermediate Representations）**を活用し、in-context learning 能力を維持しながらメロディー制御性を実現
- **Singing Voice Transcription（SVT）モジュール**：フレームレベルのピッチ・デュレーション監督を提供する補助ネットワーク
- ゼロショット学習と精密なメロディー制御を同時に達成する

### 貢献②：Coarse-to-Fine Contrastive Learning（粗から細へのコントラスト学習）

Prosody Leakage（韻律漏れ）を緩和するための新しい学習戦略として、**二段階のコントラスト学習フレームワーク**を提案する。

**Sequence-Level Contrastive Learning（シーケンスレベルのコントラスト学習）**：
- K 個の訓練サンプルが同一の歌手から収集され、同じピッチシーケンスを共有するように設定
- Semantic Acoustic Token の埋め込みを、**平均プーリング**によりグローバルに集約
- Symmetric Contrastive Loss（対称コントラスト損失、式(3)〜(4)参照）を適用し、同一歌手・同一ピッチの表現を近づけ、それ以外を遠ざける
- 目的：グローバルなメロディー形状を保持しつつ、プロンプト誘発の変動を抑制する

**Frame-Level Contrastive Learning（フレームレベルのコントラスト学習）**：
- ローカルなピッチ変動（ビブラート・グリッサンドなど）に着目
- ピッチ系列の**摂動（Perturbation）**を利用した自己教師あり学習
- フレーム単位で Pitch Token と Acoustic Token の埋め込みを整合させる
- 目的：細粒度のピッチ-音色解絡（Disentanglement）を促進し、局所的なピッチ精度を高める

この二段階コントラスト学習により、**プロンプトから音色情報のみを取得**し、ピッチ・韻律情報の混入を防ぐことができる。

### 貢献③：Singing Voice Transcription（SVT）モジュールによるメロディー監督

**Singing Voice Transcription（SVT）**（歌声転写）モジュールを補助的なピッチ監督メカニズムとして導入する。

SVTモジュールの特徴：
- **軽量 Encoder-Only Transformer**（エンコーダのみのTransformerアーキテクチャ）を採用
- 歌声音声からフレームレベルのピッチトークン系列（Pitch Token Sequence）を予測
- Pitch Vocabulary Size：12 個の離散クラス（音高クラス）× 各音高内の Attribute Token で構成
- **Stop Gradient（勾配停止）**により、S2A モデルの学習中は SVT パラメータを固定

SVT の学習目標（式(9)参照）は3つの損失の組み合わせ：
1. **Mask Prediction Loss（マスク予測損失）** L_mask：マスクされたAcoustic Tokenを雑音入力から再構成
2. **Contrastive Loss（コントラスト損失）** L_CL（シーケンスおよびフレームレベル）：メロディー条件と生成されたAcoustic Tokenの一貫性を強化
3. **SVT Loss** L_SVT：予測されたAcoustic Tokenが、SVTが推定したピッチ輪郭に韻律的・メロディー的に一致することを奨励

さらに、SVT のピッチ監督を強化するために：
- **Segment Transition Loss（セグメント遷移損失）** L_seg（式(7)）：ピッチ境界でのシャープなコントラストを促し、区間連続性を高める
- **Soft Duration Loss（ソフトデュレーション損失）** L_dur（式(8)）：各ピッチトークンに適切な確率質量を配分し、リズム的な忠実度を高める

### 貢献④：ゼロショット設定での包括的な性能向上

**Zero-Shot（ゼロショット）歌声合成**における包括的な実験結果により、CoMelSinger の有効性を示す。

主要な実験結果（Table II および Table III に基づく）：

| 指標 | CoMelSinger | 比較対象ベスト |
|------|-------------|-------------|
| MOS-Q（音質主観評価） | 3.90 | 4.10（Vevo 1.5） |
| MOS-N（自然性主観評価） | 4.00 | 4.12（Vevo 1.5） |
| SMOS（音色類似度） | 4.22 | 4.17（Vevo 1.5） |
| MCD（Mel Cepstral Distortion、メルケプストラム歪み） | 4.17 | 4.18（SPinger） |
| F0-RMSE（基本周波数誤差） | 0.042 | 0.051（Vevo 1.5） |
| SingMOS（歌唱品質自動評価） | 4.32 | 4.39（Vevo 1.5） |
| SECS（話者埋め込みコサイン類似度） | 0.912 | 0.921（GT Acoustic Codec） |

ゼロショット設定（Table III）では既存の教師あり歌声合成システムと比較可能な性能を達成し、MCD・F0-RMSE において最高値を記録している。

特筆すべき点として：
- **Timbre Consistency（音色一貫性）の高さ**：SMOS・SECSスコアから確認
- **Pitch Accuracy（ピッチ精度）の向上**：F0-RMSE において最も低い誤差を達成
- **ゼロショット汎化能力**：未見の歌手のアイデンティティを保持しながらも、提案手法の優位性を示す

---

## 4. 提案手法の核心的なアイデア

### 4-1. 全体設計思想

CoMelSinger の核心は、「**離散トークンベースの生成モデルが持つ豊かな in-context learning 能力を維持しながら、歌声合成に特有の精密なメロディー制御を実現する**」という点にある。

この二律背反を解決するために、以下の2つのアプローチを組み合わせている：

1. **Structured Conditioning（構造化条件付け）**：明示的なピッチトークンとデュレーション情報を用いて、メロディーを Semantic Token 生成に注入する
2. **Disentanglement via Contrastive Learning（コントラスト学習による解絡）**：コントラスト損失によって、Acoustic Token から音色とピッチを分離する

### 4-2. Two-Stage Discrete Token Pipeline（二段階離散トークンパイプライン）

```
[入力: 歌詞 + 音符ピッチ + プロンプト音声]
         ↓
[T2S (Text-to-Semantic) モジュール: MaskGCT ベース]
  - Semantic Acoustic Tokenizer: E_s(w) → v_Sem
  - Length Expansion: デュレーション系列 d に基づいてトークンを拡張
  - 条件付け入力: e (歌詞+音符), v_Sem (プロンプト意味論), m̂ (ピッチ整合済み)
         ↓
[Semantic Token s]
         ↓
[S2A (Semantic-to-Acoustic) モジュール: MaskGCT ベース]
  - Masked Acoustic Token 予測
  - Contrastive Loss による Timbre-Melody 解絡
  - SVT Loss による Pitch 監督
         ↓
[Acoustic Token a = Regulated Pitch Token m' + Acoustic Code]
         ↓
[Vocoder: D_A による波形再構成]
         ↓
[出力: 合成歌声]
```

### 4-3. Pitch Token の役割

CoMelSinger では、ピッチ情報を次の手順で離散トークンとして取り扱う：

1. **SVT モジュール**が音声からフレームレベルの Pitch Token 系列 m^p を予測する
2. 入力楽譜の音符情報からシンボリックなデュレーション系列 m^d を構築し、フレームレベルに正規化する（式(2)参照）
3. 正規化した Pitch Token m̂ を T2S モジュールの条件付け入力として組み込む
4. S2A 段階では、SVT の StopGradient を経た m^p を参照し、SVT Loss で生成 Acoustic Token を誘導する

この設計により、ピッチ情報が Semantic・Acoustic の両段階で一貫して埋め込まれる。

### 4-4. Contrastive Learning の設計

**Sequence-Level Contrastive Loss（式(3)〜(4)）**：
- バッチ内で同一歌手・同一ピッチの K 個のサンプルを Positive Pair として定義
- Prompt と非プロンプト（対角外）を Negative Pair とする対称損失
- グローバルな音色表現を強制的に整合させ、プロンプトのメロディー依存を遮断する

**Frame-Level Contrastive Loss（式(5)〜(6)）**：
- 同一サンプルに対してピッチ摂動（Perturbation）を加えた版を生成し、Positive Pair とする
- 対角外のフレームペアを Negative とする
- ローカルなピッチ変動パターンを学習しつつ、音色情報とピッチ情報を解絡する

### 4-5. 推論時の動作

推論時（Inference）には：
- 並列な非自己回帰（NAR）デコーディングで高速生成
- 外部の音声プロンプトのみを必要とし、未見の歌手に対してゼロショット合成が可能
- 音符・歌詞の入力があれば任意のメロディーを精密に制御可能

---

## 5. 論文の構成

| セクション | タイトル | 内容 |
|-----------|--------|------|
| Abstract（抄録） | — | 問題設定・提案手法・実験結果の概要 |
| I. Introduction（序論） | Introduction | SVSの背景・Prosody Leakageの問題・本論文の位置付け |
| II. Related Work（関連研究） | Related Work | A. Singing Voice Synthesis / B. Prosody and Melody Control |
| III. Method（手法） | Method | A. Overview / B. T2S Stage / C. Coarse-to-Fine Contrastive Learning / D. Singing Voice Transcription for Pitch Guidance / E. Training and Inference Procedures |
| IV. Experimental Setups（実験設定） | Experimental Setups | A. Dataset / B. Seen Singer Evaluation / C. Zero-Shot Singing Voice Synthesis |
| V. Experimental Results（実験結果） | Experimental Results | A. MaskGCT の Prosody Similarity 評価 / B. Seen Singer 評価 / C. Zero-Shot 評価 / D. Ablation Study（消去実験） |
| VI. Conclusion（結論） | Conclusion | まとめと今後の展望 |
| References（参考文献） | — | 75件の参考文献 |

### 各セクションの詳細

**Section III: Method（手法）の詳細構成**
- **III-A. Overview**：二段階パイプラインの全体像と各モジュールの役割
- **III-B. T2S Stage**：テキスト・音符・プロンプトから Semantic Token を生成する段階の設計
- **III-C. Coarse-to-Fine Contrastive Learning**：
  - Sequence-Level Symmetric Contrastive Loss の設計
  - Frame-Level Contrastive Loss の設計
- **III-D. Singing Voice Transcription for Pitch Guidance**：
  - SVT モジュールのアーキテクチャ
  - Soft Duration Loss と Segment Transition Loss
  - SVT Loss の総合的な定式化（式(9)）
- **III-E. Training and Inference Procedures**：
  - SVT モデルの事前学習
  - S2A モデルのファインチューニング（Algorithm 1）
  - 推論手順

**Section V: Experimental Results（実験結果）の詳細構成**
- **V-A. MaskGCT の Prosody Leakage 評価**：LibriTTS と AISHELL-3 でプロンプト誘発の韻律類似度を分析（Table I）
- **V-B. Seen Singer 評価**：主観（MOS）・客観（MCD, F0-RMSE, SingMOS, SECS）両指標での比較（Table II）
- **V-C. Zero-Shot 評価**：未見の歌手への汎化性能評価（Table III）と零ショット専用ベースラインとの比較（Table IV）
- **V-D. Ablation Study（消去実験）**：
  - コンポーネント分析（Table V）：CL・SVT の各要素を個別に除去して効果を検証
  - Fine-Tuning Strategy の比較（Table VI・VII）：SVT のファインチューニング手法の検討

---

## 補足：評価指標の説明

| 指標 | 英語名（略称） | 日本語説明 |
|------|-------------|----------|
| MOS-Q | Mean Opinion Score - Quality | 音質の主観評価スコア（5点満点） |
| MOS-N | Mean Opinion Score - Naturalness | 自然性の主観評価スコア（5点満点） |
| SMOS | Singing Mean Opinion Score | 音色類似度の主観評価スコア（5点満点） |
| MCD | Mel Cepstral Distortion | メルケプストラム歪み（音質の客観指標、低いほど良い） |
| F0-RMSE | Fundamental Frequency Root Mean Square Error | 基本周波数の二乗平均平方根誤差（低いほど良い） |
| SingMOS | Singing MOS（自動評価） | 大規模歌唱データで訓練された知覚品質の自動評価指標 |
| SECS | Speaker Embedding Cosine Similarity | 話者埋め込みコサイン類似度（音色同一性の指標） |

---

*本ドキュメントは arXiv:2509.19883v2（2025年4月11日版）に基づいて作成された。*
