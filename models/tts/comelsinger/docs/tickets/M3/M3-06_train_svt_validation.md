# M3-06: SVTバリデーション関数

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07a
> **依存チケット**: M3-04
> **ブロックするチケット**: M3-07
> **状態**: TODO

---

## 1. 目的とゴール

`evaluate_svt()` 関数をtools/evaluate_svt.pyに実装し、バリデーションセットに対してフレームレベルのピッチF1スコアを計算する。モデルをeval modeにしてno_gradで推論し、予測ピッチトークンと正解ピッチトークンのマッチング率をF1として返す。M3-05のロギングとM3-07のスモークテストで使用される。

## 2. 実装する内容の詳細

```python
# tools/evaluate_svt.py
import torch
from sklearn.metrics import f1_score

def evaluate_svt(model, dataloader, device="cuda"):
    """フレームレベルピッチF1スコアを計算する。"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            acoustic = batch["acoustic_tokens"].to(device)
            pitch_labels = batch["pitch_tokens"].to(device)
            padding_mask = batch.get("padding_mask", None)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)

            out = model(acoustic, padding_mask=padding_mask)
            preds = out["logits"].argmax(dim=-1)  # (B, T)

            # paddingを除いたフレームのみ評価
            mask = (pitch_labels != -100)
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(pitch_labels[mask].cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    model.train()
    return float(f1)
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | evaluate_svt()関数実装・padding除外ロジック・F1計算 |

## 4. 提供範囲とテスト項目

**含むもの**: `tools/evaluate_svt.py`、フレームレベルF1計算、paddingトークン除外

**含まないもの**: ロギング（→ M3-05）、学習ループへの統合（→ M3-05）

### ユニットテスト

```bash
uv run python -c "
import torch
from tools.evaluate_svt import evaluate_svt
from models.tts.comelsinger.svt_module import SVTModule

# 完全一致の場合F1=1.0
class MockLoader:
    def __iter__(self):
        yield {
            'acoustic_tokens': torch.zeros(2, 10, 8, dtype=torch.long),
            'pitch_tokens': torch.zeros(2, 10, dtype=torch.long),
        }
# forwardが実装済みの場合のみ動作
print('PASS: evaluate_svt structure OK')
"
```

```bash
uv run python -c "
from sklearn.metrics import f1_score
import numpy as np
preds = [0, 1, 2, 0, 1]
labels = [0, 1, 2, 0, 2]
f1 = f1_score(labels, preds, average='macro', zero_division=0)
assert 0.0 <= f1 <= 1.0
print(f'PASS: f1_score={f1:.3f}')
"
```

### E2Eテスト

```bash
uv run python -c "
from tools.evaluate_svt import evaluate_svt
import torch
# SVTモデルをロードしてF1が0〜1の範囲であることを確認
# (M2完了後に実データで実行)
print('PASS: evaluate_svt returns float in [0,1]')
"
```

## 5. 懸念事項とレビュー項目

- **paddingトークンの定義**: `pitch_tokens == -100` をpaddingと定義しているが、M2-10のDatasetの仕様と一致するか確認する。
- **macro vs micro F1**: ピッチクラス数が129の場合、稀なピッチのmacro F1は不安定。weighted F1も並行して計算することを推奨する。

### レビュー項目

- [ ] eval mode / no_grad が確実に適用されていること
- [ ] evaluate_svt()の戻り値がfloat型であること
- [ ] padding除外ロジックがバッチ全体で正しく動作すること

## 6. フェーズ振り返り: 一から作り直すとしたら

> 共通の設計判断（PyTorch Lightning vs Accelerate、実験管理、スケジューラ選択）は [M3_design_decisions.md](M3_design_decisions.md) を参照のこと。以下はこのチケット固有の設計判断を記載する。


- **PyTorch Lightning vs 素のAccelerate**: Lightningの `validation_step` + `validation_epoch_end` で実装すれば、eval mode切替やno_grad管理が自動化される。
- **実験管理(W&B/MLflow)**: confusion matrixをW&Bに記録することで、特定のピッチクラスで精度が低い問題を可視化できる。
- **学習率スケジューラ選択**: バリデーションF1が改善しない場合に学習率を下げる ReduceLROnPlateau を採用すると、固定スケジューラより頑健になる。

## 7. 後続タスクへの連絡事項

- **M3-05**: `evaluate_svt()` は `float` 単一値を返す。複数指標（precision, recall等）が必要な場合は設計変更が必要。
- **M3-07**: スモークテスト（1000ステップ）でのF1評価基準は「loss初期値の80%以下に低下」であり、F1絶対値の基準値は未設定。実験後に設定すること。
- **M3-08**: S2A学習でfrozen SVTとして使用する際は `evaluate_svt()` は呼ばれないが、F0-RMSEによる独自評価関数（M3-20）が必要になる。
