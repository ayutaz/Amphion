# M4-09: SVT F1 評価スクリプト

> **マイルストーン**: [M4: 推論・評価](../../13_milestones.md#m4-推論評価システム)
> **対応RQ**: RQ-09
> **依存チケット**: M4-04
> **ブロックするチケット**: M4-11, M4-13
> **状態**: TODO

---

## 1. 目的とゴール

合成音声に対して frozen SVT モジュールでピッチ予測を実行し、グラウンドトゥルースのピッチトークンとの F1 スコアを計算するスクリプトを実装する。パイプラインは「合成音声 → EnCodec で音響トークン → frozen SVT → ピッチトークン予測 → GT ピッチトークンと比較」の順に処理する。論文の目標値は **F1 = 0.711**。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/eval/eval_svt_f1.py
from models.tts.comelsinger.svt_module import SVTModule
from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer
import torch
from sklearn.metrics import f1_score

def compute_svt_f1(
    syn_acoustic_tokens: torch.Tensor,  # (T, 8) codec tokens
    gt_pitch_tokens: torch.Tensor,      # (T,) int64
    svt_model: SVTModule,
    device="cuda",
) -> float:
    svt_model.eval()
    with torch.no_grad():
        pred_logits = svt_model(syn_acoustic_tokens.to(device))  # (T, n_bins)
        pred_tokens = pred_logits.argmax(dim=-1).cpu().numpy()
    gt = gt_pitch_tokens.cpu().numpy()
    # フレーム単位の macro F1（パディングトークンを除外）
    mask = gt != 0  # 0 はパディング
    return float(f1_score(gt[mask], pred_tokens[mask], average="macro"))
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `compute_svt_f1()` 関数・SVT ロード処理実装 |
| 検証エージェント | 1 | GT ピッチトークンの生成処理・パディング除外ロジック確認 |

## 4. 提供範囲とテスト項目

**含むもの**: `compute_svt_f1()` 関数、frozen SVT ロード、50 発話一括評価 CLI、JSON 出力

**含まないもの**: SVT モデル自体の学習（M3 の範囲）、F0-RMSE との比較分析

### ユニットテスト

```python
def test_perfect_prediction_gives_one(svt_model):
    # 予測が GT と完全一致する場合、F1=1.0
    tokens = torch.randint(1, 128, (100,))
    # SVT が常に GT を返すモックを使用
    score = compute_svt_f1_with_mock(tokens, tokens)
    assert abs(score - 1.0) < 1e-5

def test_padding_excluded_from_f1(svt_model):
    gt = torch.tensor([0, 0, 60, 62, 64, 0])  # 0 はパディング
    # パディングを含めずに F1 を計算することを確認
    score = compute_svt_f1(dummy_acoustic, gt, svt_model)
    assert 0.0 <= score <= 1.0
```

### E2Eテスト

```bash
uv run python models/tts/comelsinger/eval/eval_svt_f1.py \
  --syn_dir outputs/seen_results/ \
  --gt_tokens_dir data/gt_pitch_tokens_seen/ \
  --svt_ckpt checkpoints/svt.pt \
  --output results/svt_f1_seen.json
# → {"mean_f1": 0.711, "std": 0.05, "n_samples": 50}
```

## 5. 懸念事項とレビュー項目

- **合成音声 → 音響トークン**: EnCodec encoder を通す処理が eval スクリプト内に必要。M4-04 の出力（WAV）から音響トークンを再エンコードする処理が重複しないよう設計すること。
- **SVT の frozen 状態**: 評価時に SVT の重みが更新されないよう `torch.no_grad()` と `model.eval()` を必ず使用すること。

レビュー項目:
- [ ] SVT モデルの重みロード先（`svt_ckpt`）が M3 の出力チェックポイントと一致しているか
- [ ] `macro` F1 と `micro` F1 のどちらを使うか（論文の定義を確認）
- [ ] 出力 JSON のスキーマが M4-13 の期待する形式と一致しているか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **評価パイプライン自動化（CI 統合）**: SVT F1 を学習チェックポイントごとに自動計算し、ピッチ制御精度の推移をトラッキングする CI ジョブを最初から設計する。
- **主観評価プラットフォーム選定**: SVT F1（自動）と SMOS（主観メロディ類似度）の相関を検証し、自動指標の妥当性を評価する。
- **Ablation 自動実行**: `wo_svt` 条件（SVT なし）との F1 差分を自動計算し、SVT の効果を定量化するスクリプトを最初から用意する。

## 7. 後続タスクへの連絡事項

- M4-11 はこの JSON 出力（`svt_f1_seen.json`, `svt_f1_unseen.json`）を読み込む。`mean_f1` と `std` を必ず含めること。
- M4-13 は SVT F1 を論文 Table III 形式に整形する。ablation 条件（M4-10）ごとの F1 値も計算が必要なため、M4-10 担当者と連携すること。
- SVT チェックポイント（`svt.pt`）のパスを M4-10 の ablation 設定 YAML に記載すること。
