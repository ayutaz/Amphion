# M3-07: SVTスモークテスト

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07a
> **依存チケット**: M3-05, M3-06
> **ブロックするチケット**: M3-08
> **状態**: TODO

---

## 1. 目的とゴール

batch=4・1000ステップの縮小実験を実行し、学習終了時の損失値が初期値の80%以下に低下することを確認する。スモークテストはS2A学習（M3-08）に移行する前の品質ゲートとして機能する。テスト用の小規模データセット（100サンプル程度）を使用する。

## 2. 実装する内容の詳細

```python
# tools/smoke_test_svt.py
import subprocess, sys, json

def run_smoke_test():
    """1000ステップのSVTスモークテストを実行し、損失低下を検証する。"""
    result = subprocess.run([
        sys.executable, "tools/train_svt.py",
        "--config", "configs/comelsinger/svt_train.yaml",
        "--max-steps", "1000",
        "--batch-size", "4",
        "--log-dir", "runs/smoke_svt",
        "--save-loss-history", "runs/smoke_svt/loss_history.json",
    ], capture_output=True, text=True)
    assert result.returncode == 0, f"Training failed: {result.stderr}"

    with open("runs/smoke_svt/loss_history.json") as f:
        history = json.load(f)
    init_loss = history["losses"][0]
    final_loss = history["losses"][-1]
    ratio = final_loss / init_loss
    assert ratio < 0.8, f"Loss did not decrease enough: {init_loss:.3f} -> {final_loss:.3f} (ratio={ratio:.2f})"
    print(f"PASS: loss {init_loss:.3f} -> {final_loss:.3f} (ratio={ratio:.2f} < 0.80)")
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| テスト実行エージェント | 1 | スモークテストスクリプト作成・実行・結果記録 |

## 4. 提供範囲とテスト項目

**含むもの**: 1000ステップ学習実行、損失履歴記録、80%低下基準の検証スクリプト

**含まないもの**: 本番50Kステップ学習（-> 本番環境での実行）、S2A学習（-> M3-08以降）

### ユニットテスト

```bash
uv run python -c "
# 損失低下判定ロジックの単体テスト
init_loss, final_loss = 2.5, 1.8
ratio = final_loss / init_loss
assert ratio < 0.8, f'ratio={ratio}'
print(f'PASS: loss decrease check OK (ratio={ratio:.2f})')
"
```

```bash
uv run python -c "
import json, os
os.makedirs('runs/smoke_svt', exist_ok=True)
# loss_history.jsonのフォーマット確認
history = {'losses': [2.5, 2.3, 2.0, 1.9, 1.8], 'steps': [1, 200, 400, 600, 1000]}
with open('runs/smoke_svt/loss_history.json', 'w') as f:
    json.dump(history, f)
print('PASS: loss_history.json format OK')
"
```

### E2Eテスト

```bash
uv run python tools/smoke_test_svt.py
# PASS: loss X.XXX -> X.XXX (ratio=X.XX < 0.80) が出力されること
```

## 5. 懸念事項とレビュー項目

- **80%基準の妥当性**: 1000ステップでの期待損失低下は過学習を確認するものでなく、学習ループが正常に機能しているかの確認。基準が厳しすぎる場合は90%に緩和を検討する。
- **縮小データセット**: 本番データ（M4Singer 29時間）がない場合、生成したダミーデータでのスモークテストを先行して実施する。

### レビュー項目

- [ ] 1000ステップ後の損失が初期値の80%以下であること
- [ ] 学習中にNaN/Infが発生していないこと
- [ ] スモークテスト結果がファイルに記録されること

## 6. フェーズ振り返り: 一から作り直すとしたら

> 共通の設計判断（PyTorch Lightning vs Accelerate、実験管理、スケジューラ選択）は [M3_design_decisions.md](M3_design_decisions.md) を参照のこと。以下はこのチケット固有の設計判断を記載する。


- **PyTorch Lightning vs 素のAccelerate**: LightningのTrainerは `fast_dev_run=True` でスモークテスト相当の機能を内蔵する。専用スクリプトが不要になる。
- **実験管理(W&B/MLflow)**: スモークテストの結果をW&Bに自動記録し、本番実験と比較できる仕組みを最初から構築する。
- **学習率スケジューラ選択**: 1000ステップ程度では CosineAnnealingLR のウォームアップ段階のみ。ウォームアップ付きスケジューラ（LinearWarmup + Cosine）での確認が必要。

## 7. 後続タスクへの連絡事項

- **M3-08**: SVT学習が収束することを確認してからS2A学習に進む。`checkpoints/svt_best.pt` の存在を前提とする。
- **M3-21**: S2Aのスモークテスト（5epoch, batch=4）も本チケットと同様の構成で実施する。SVTスモークテストのスクリプト設計を参考にすること。
- **M3-22**: スモークテストの結果（損失値、F1スコア）を設定値クロスチェックのエビデンスとして記録しておくこと。
