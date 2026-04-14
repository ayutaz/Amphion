# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoMelSinger の再現実装プロジェクト。Amphion フレームワーク（MaskGCT）を fork し、歌唱音声合成（SVS）向けの拡張を行う。

**対象論文**: "CoMelSinger: Discrete Token-Based Zero-Shot Singing Synthesis With Structured Melody Control and Guidance" (Zhao et al., 2026, arXiv:2509.19883v2)
**論文PDF**: `/Users/inamotoyuuta/Downloads/2509.19883v2.pdf`
**Fork元**: https://github.com/open-mmlab/Amphion
**Fork先**: https://github.com/ayutaz/Amphion

## Architecture

### 全体構造（2段階非自己回帰フレームワーク）

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
│  Stage 2: S2A (Semantic-to-Acoustic)│  ★ ここを拡張
│  + ピッチ埋め込み条件付け             │
│  + Contrastive Learning (SCL+FCL)   │
│  + SVT ピッチ監督                    │
└───────────────┬─────────────────────┘
                ▼
┌─────────────────────────────────────┐
│  EnCodec Decoder（事前学習済み）      │
│  音響トークン(RVQ 8層) → 波形         │
└─────────────────────────────────────┘
```

### MaskGCT ベースコード（既存・変更最小限）

- `models/tts/maskgct/maskgct_s2a.py` — MaskGCT_S2A クラス: DiffLlama backbone, RVQ 8層マスク予測, forward_diffusion/loss_t メソッド
- `models/tts/maskgct/maskgct_t2s.py` — MaskGCT_T2S クラス: DiffLlamaPrefix backbone, セマンティックトークン生成
- `models/tts/maskgct/llama_nar.py` — DiffLlama/DiffLlamaPrefix: LlamaDecoderLayer + LlamaAdaptiveRMSNorm, SinusoidalPosEmb
- `models/tts/maskgct/maskgct_utils.py` — Codec/semantic extraction utilities
- `models/tts/maskgct/maskgct_inference.py` — 推論パイプライン（反復デコーディング）

### CoMelSinger 固有モジュール（`models/tts/comelsinger/` に新規実装）

| モジュール | 概要 | 論文参照 |
|---|---|---|
| **CoMelSinger_S2A** | MaskGCT_S2A を継承・拡張。ピッチ埋め込み層を追加 | Section III-A |
| **SVT Module** | 軽量 encoder-only Transformer（4層, hidden 512, heads 8）。音響トークン→ピッチトークン予測 | Section III-C |
| **Contrastive Loss** | SCL（シーケンスレベル）+ FCL（フレームレベル）。prosody leakage 防止 | Section III-B |
| **Pitch Tokenizer** | F0 → 離散ピッチトークン（MIDI量子化） | Section III-C |
| **Soft Duration Loss** | ピッチトークンの時間配分を監督 | 数式(8) |

### 損失関数の構成（S2A学習時）

```
L = λ_CL · L_CL + λ_SVT · L_SVT + λ_mask · L_mask

L_CL  = λ_SCL · L_SCL + λ_FCL · L_FCL   (対照学習)
L_SVT = L_CE + λ_seg · L_seg + λ_dur · L_dur  (SVTからのピッチ監督、frozen)
L_mask = MaskLoss(â, a)                    (MaskGCTのマスク予測損失)

重み: λ_SCL=0.5, λ_FCL=1.0, λ_SVT=0.1, λ_mask=0.3, λ_seg=3, λ_dur=5
```

### 事前学習済みモデル（HuggingFace）

- `amphion/MaskGCT-T2S` — T2Sモデルの初期重み
- `amphion/MaskGCT-S2A` — S2Aモデルの初期重み（ここにLoRAを適用してfine-tune）

## Key Design Decisions

- 非自己回帰生成（MaskGCTパラダイム）で並列デコーディングとグローバル文脈を実現
- ピッチトークンは音響トークンから分離し、明示的メロディ制御を可能にする
- 対照学習で参照音声からの prosody leakage（韻律漏れ）を防止
- SVTモジュールはS2A学習時に frozen（StopGrad）で固定ピッチ監督として機能
- FCLでは soft label matrix（hard labelでなく）を使用し、ピッチのパディング/繰り返しに対応

## Datasets

- **M4Singer**: 中国語ポップス、20話者、約29時間（CC BY-NC-SA 4.0）
- **Opencpop**: 中国語ポップス、1話者、約5.2時間（研究目的限定）
- **MIR-ST500**: 500曲、160k+注釈ノート（SVT汎化評価用）

## Hardware Requirements

- SVT学習: NVIDIA RTX A5000 × 1（24GB VRAM）
- S2A fine-tuning: NVIDIA RTX A5000 × 4（24GB × 4）
- LoRA で全パラメータの約4.8%のみ更新

## Tech Stack

- Python (PyTorch 2.1+), transformers (LlamaConfig/LlamaModel)
- Audio codec: EnCodec (`facebook/encodec_24khz`, 8 RVQ codebooks, 24kHz)
- Fine-tuning: LoRA via HuggingFace PEFT (`r=16, alpha=32, target=q_proj,v_proj`)
- Semantic features: Whisper encoder / wav2vec 2.0
- Pitch extraction: pyworld/crepe → F0 → discrete pitch tokens
- G2P: pypinyin（中国語テキスト→ピンイン）

## Development Notes

- Primary language for documentation: Japanese（ドキュメントは全て日本語）
- Code comments and variable names: English
- 論文分析ドキュメント（10本）: `models/tts/comelsinger/docs/`
- 論文PDF: `/Users/inamotoyuuta/Downloads/2509.19883v2.pdf`

## Implementation Status

- [x] 論文分析ドキュメント作成（10本、合計5,332行）
- [x] Amphion fork & sparse-checkout
- [x] プロジェクト構造セットアップ
- [ ] ピッチトークナイザー実装
- [ ] SVTモジュール実装
- [ ] S2A拡張（ピッチ埋め込み + 対照学習）
- [ ] 学習パイプライン
- [ ] 推論パイプライン
- [ ] データ前処理パイプライン
- [ ] 評価スクリプト
