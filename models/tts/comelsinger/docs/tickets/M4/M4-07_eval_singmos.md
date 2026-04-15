# M4-07: SingMOS 評価スクリプト

> **マイルストーン**: [M4: 推論・評価](../../13_milestones.md#m4-推論評価システム)
> **対応RQ**: RQ-09
> **依存チケット**: M4-04
> **ブロックするチケット**: M4-11, M4-13
> **状態**: TODO

---

## 1. 目的とゴール

歌唱音声の主観品質を自動推定する SingMOS モデルを使い、生成音声の歌唱品質スコアを計算するスクリプトを実装する。South-Twilight/SingMOS（HuggingFace）を使用し、50 発話のスコアを平均して報告する。Seen セットの目標値は **4.32**。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/eval/eval_singmos.py
from transformers import AutoProcessor, AutoModelForAudioClassification
import torch, numpy as np, soundfile as sf

SINGMOS_MODEL_ID = "South-Twilight/SingMOS"

def load_singmos(device="cuda"):
    processor = AutoProcessor.from_pretrained(SINGMOS_MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(SINGMOS_MODEL_ID).to(device)
    model.eval()
    return processor, model

def compute_singmos(wav: np.ndarray, sr: int,
                    processor, model, device="cuda") -> float:
    inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    # SingMOS は回帰出力（スカラー MOS スコア）
    return float(logits.squeeze().cpu())
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `load_singmos()` / `compute_singmos()` 実装 |
| 検証エージェント | 1 | SingMOS モデルの出力範囲・精度確認 |

## 4. 提供範囲とテスト項目

**含むもの**: `compute_singmos()` 関数、50 発話一括評価 CLI、JSON 出力

**含まないもの**: SingMOS モデル自体の学習・fine-tuning、他 MOS 指標（MOS-Q/N → M4-12）

### ユニットテスト

```python
def test_singmos_score_in_range(processor, model):
    wav = np.random.randn(24000).astype(np.float32)
    score = compute_singmos(wav, 24000, processor, model, device="cpu")
    assert 1.0 <= score <= 5.0

def test_singmos_deterministic(processor, model):
    wav = np.random.randn(24000).astype(np.float32)
    s1 = compute_singmos(wav, 24000, processor, model, device="cpu")
    s2 = compute_singmos(wav, 24000, processor, model, device="cpu")
    assert abs(s1 - s2) < 1e-5
```

### E2Eテスト

```bash
uv run python models/tts/comelsinger/eval/eval_singmos.py \
  --syn_dir outputs/seen_results/ \
  --output results/singmos_seen.json
# → {"mean_singmos": 4.32, "std": 0.15, "n_samples": 50}
```

## 5. 懸念事項とレビュー項目

- **SingMOS モデルの入力仕様**: `South-Twilight/SingMOS` が要求するサンプリングレート・長さ制約を確認する。24kHz 以外を要求する場合はリサンプリングが必要。
- **モデルの出力形式**: 分類モデルか回帰モデルかを確認し、出力を 1〜5 のスコアに正規化する処理を実装すること。

レビュー項目:
- [ ] SingMOS の出力が 1〜5 の範囲に収まっているか
- [ ] GPU がない環境（CPU）でも動作するか
- [ ] 出力 JSON のスキーマが M4-13 の期待する形式と一致しているか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **評価パイプライン自動化（CI 統合）**: SingMOS をバッチ処理で GPU フル活用するよう最適化し、CI の評価ステップに組み込む。
- **主観評価プラットフォーム選定**: SingMOS（自動）と MOS-Q（主観）の相関を分析するフローを評価設計段階で組み込み、自動評価の信頼性を定量化する。
- **Ablation 自動実行**: 6 条件の SingMOS を一括計算するラッパースクリプトを最初から用意する。

## 7. 後続タスクへの連絡事項

- M4-11 はこの JSON 出力（`singmos_seen.json`, `singmos_unseen.json`）を読み込む。`mean_singmos` と `std` を必ず含めること。
- M4-12 の主観評価（MOS-Q/N/SMOS）と SingMOS スコアの相関分析を M4-13 で実施する。このチケット完了後に M4-12 担当者と数値を突き合わせること。
- `South-Twilight/SingMOS` のライセンスを確認し、研究目的での使用が許可されていることを記録しておくこと。
