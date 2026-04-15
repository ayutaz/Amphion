# M4-08: SECS（話者類似度）評価スクリプト

> **マイルストーン**: [M4: 推論・評価](../../13_milestones.md#m4-推論評価システム)
> **対応RQ**: RQ-09
> **依存チケット**: M4-04
> **ブロックするチケット**: M4-11, M4-13
> **状態**: TODO

---

## 1. 目的とゴール

参照音声と生成音声の話者類似度（SECS: Speaker Embedding Cosine Similarity）を計算するスクリプトを実装する。`microsoft/wavlm-base-plus` で話者埋め込みを抽出し、コサイン類似度を算出する。論文の目標値は **0.912**（Seen セット）。スクリプトは `eval_secs.py` として独立実行可能であり、JSON 形式でスコアを出力する。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/eval/eval_secs.py
from transformers import WavLMModel, AutoFeatureExtractor
import torch, torch.nn.functional as F, numpy as np

WAVLM_MODEL_ID = "microsoft/wavlm-base-plus"

def extract_speaker_embedding(wav: np.ndarray, sr: int,
                               feature_extractor, model, device="cuda") -> torch.Tensor:
    inputs = feature_extractor(wav, sampling_rate=sr, return_tensors="pt").to(device)
    with torch.no_grad():
        # 最終隐層の平均プーリングで話者埋め込みを取得
        hidden = model(**inputs).last_hidden_state  # (1, T, D)
    return hidden.mean(dim=1)  # (1, D)

def compute_secs(ref_wav, syn_wav, sr, feature_extractor, model, device) -> float:
    ref_emb = extract_speaker_embedding(ref_wav, sr, feature_extractor, model, device)
    syn_emb = extract_speaker_embedding(syn_wav, sr, feature_extractor, model, device)
    return float(F.cosine_similarity(ref_emb, syn_emb).cpu())
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `extract_speaker_embedding()` / `compute_secs()` 実装 |
| 検証エージェント | 1 | 同話者ペアの類似度 > 0.9 であることを既知データで確認 |

## 4. 提供範囲とテスト項目

**含むもの**: `compute_secs()` 関数、50 発話一括評価 CLI、JSON 出力

**含まないもの**: WavLM モデル自体の fine-tuning、他の話者埋め込みモデルへの切り替え

### ユニットテスト

```python
def test_same_wav_gives_high_secs(feature_extractor, model):
    wav = np.random.randn(24000).astype(np.float32)
    score = compute_secs(wav, wav, 24000, feature_extractor, model, "cpu")
    assert score > 0.99  # 完全一致 ≈ 1.0

def test_different_wavs_lower_secs(feature_extractor, model):
    ref = np.random.randn(24000).astype(np.float32)
    syn = np.random.randn(24000).astype(np.float32)
    score = compute_secs(ref, syn, 24000, feature_extractor, model, "cpu")
    assert score < 0.99
```

### E2Eテスト

```bash
uv run python models/tts/comelsinger/eval/eval_secs.py \
  --ref_dir data/gt_seen/ \
  --syn_dir outputs/seen_results/ \
  --output results/secs_seen.json
# → {"mean_secs": 0.912, "std": 0.02, "n_samples": 50}
```

## 5. 懸念事項とレビュー項目

- **WavLM の話者埋め込み層**: `last_hidden_state` の平均プーリングではなく、特定の中間層を使う実装もある。論文が使用した層を Section IV-C から確認すること。
- **入力長の制限**: WavLM は非常に長い音声でメモリ不足になる場合がある。最大長を設定し、超過する場合は先頭の N 秒を使うようにすること。

レビュー項目:
- [ ] コサイン類似度の計算が `F.cosine_similarity` で正しく実装されているか（次元指定）
- [ ] `ref_audio` がテストセット JSON の `ref_audio_path` から正しく読み込まれているか
- [ ] 出力 JSON のスキーマが M4-13 の期待する形式と一致しているか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **評価パイプライン自動化（CI 統合）**: 学習チェックポイントごとに SECS を自動計算して話者類似度の推移をグラフ化する CI ジョブを最初から設計する。
- **主観評価プラットフォーム選定**: SECS（自動）と MOS-N（主観自然性）の相関を分析し、自動評価の代理指標としての妥当性を検証するフローを組み込む。
- **Ablation 自動実行**: prosody leakage の程度（音色 vs 韻律の分離）を SECS で定量化する ablation を最初から計画する。

## 7. 後続タスクへの連絡事項

- M4-11 はこの JSON 出力（`secs_seen.json`, `secs_unseen.json`）を読み込む。`mean_secs` と `std` を必ず含めること。
- M4-13 は SECS を論文 Table II 形式に整形する。Seen 0.912 / Unseen 両方の値が必要。
- `microsoft/wavlm-base-plus` のダウンロードサイズは約 360 MB。ネットワーク環境が制限されている場合は事前にキャッシュしておくこと（`HF_HOME` 環境変数でキャッシュ先を指定）。
