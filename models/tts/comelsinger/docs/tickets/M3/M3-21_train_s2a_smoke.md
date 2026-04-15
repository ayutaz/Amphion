# M3-21: S2Aスモークテスト

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07b
> **依存チケット**: M3-20
> **ブロックするチケット**: M3-22
> **状態**: TODO

---

## 1. 目的とゴール

5epoch・batch=4の縮小実験を実行し、全5損失（L_SCL, L_FCL, L_CL, L_SVT, L_mask）がNaN/Infを出さないことを確認する。S2A学習パイプラインの全コンポーネントが正常に動作することを最終確認するゲートチェック。

## 2. 実装する内容の詳細

```python
# tools/smoke_test_s2a.py

import subprocess, sys, json
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def run_s2a_smoke_test():
    """S2Aの5エポックスモークテストを実行し、全損失がNaN/Infでないことを確認する。"""
    result = subprocess.run([
        sys.executable, "tools/train_s2a.py",
        "--config", "configs/comelsinger/s2a_train.yaml",
        "--max-epochs", "5",
        "--batch-size", "4",
        "--log-dir", "runs/smoke_s2a",
        "--ckpt-dir", "checkpoints/smoke_s2a",
    ], capture_output=True, text=True)
    assert result.returncode == 0, f"Training failed:\n{result.stderr}"

    ea = EventAccumulator("runs/smoke_s2a"); ea.Reload()
    for key in ["train/l_scl", "train/l_fcl", "train/l_cl",
                "train/l_svt", "train/l_mask", "train/l_total"]:
        values = [s.value for s in ea.Scalars(key)]
        assert all(not (v != v or abs(v) == float("inf")) for v in values), \
            f"NaN/Inf detected in {key}: {values}"
    print("PASS: all 6 loss values are finite across 5 epochs")
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| テスト実行エージェント | 1 | smoke_test_s2a.py作成・実行・全損失NaN/Inf確認 |

## 4. 提供範囲とテスト項目

**含むもの**: 5epoch・batch=4のS2A学習実行、全損失のNaN/Inf検証スクリプト

**含まないもの**: 本番100epoch学習（→ 本番環境での実行）、推論評価（→ M4）

### ユニットテスト

```bash
uv run python -c "
import math
values = [2.5, 2.3, 2.1, 1.9, 1.8]
assert all(not math.isnan(v) and not math.isinf(v) for v in values)
print('PASS: NaN/Inf check logic OK')
"
```

```bash
uv run python -c "
import torch
# 学習後の損失テンソルがfiniteであることを確認するユーティリティ
def check_finite(loss_dict):
    return all(torch.isfinite(torch.tensor(v)).item() for v in loss_dict.values())
losses = {'l_scl': 1.2, 'l_fcl': 0.8, 'l_cl': 1.28, 'l_svt': 0.5, 'l_mask': 2.1}
assert check_finite(losses)
print('PASS: finite check utility OK')
"
```

### E2Eテスト

```bash
uv run python tools/smoke_test_s2a.py
# PASS: all 6 loss values are finite across 5 epochs が出力されること
```

## 5. 懸念事項とレビュー項目

- **バッチサイズ4でのK_s=8**: batch=4はK_s=8より小さいため、split_batch()がエラーを起こす可能性がある。スモークテスト用に `K_s=min(batch_size//2, K_s)` の保護を追加する。
- **S2A初回forwardでのNaN**: LoRA初期化直後はピッチ埋め込み層の初期値によってNaNが発生することがある。初期化にkaiming_normalを使用することを確認する。

### レビュー項目

- [ ] 5epoch完了後に全損失がNaN/Infでないこと
- [ ] スモークテストがbatch=4で動作すること（K_s処理を含む）
- [ ] TensorBoardのeventファイルが生成されること

## 6. フェーズ振り返り: 一から作り直すとしたら

> 共通の設計判断（PyTorch Lightning vs Accelerate、実験管理、スケジューラ選択）は [M3_design_decisions.md](M3_design_decisions.md) を参照のこと。以下はこのチケット固有の設計判断を記載する。


- **PyTorch Lightning vs 素のAccelerate**: Lightningでは `fast_dev_run=5` で5バッチのみ実行するスモークテストが内蔵されており、専用スクリプトが不要になる。
- **実験管理(W&B/MLflow)**: スモークテストの結果をW&Bで本番実験と比較できるよう、実験名に `smoke_` プレフィックスをつけて記録する仕組みを最初から設計する。
- **GradNorm動的重み調整**: スモークテスト段階で各損失の相対的なgrad normを計測し、本番実験でのGradNorm設定値を決定するための情報収集に活用できる。

## 7. 後続タスクへの連絡事項

- **M3-22**: スモークテストの結果（全損失がfinite）がパスしてからM3-22のクロスチェックに進むこと。スモークテストの結果をM3-22の確認エビデンスとして記録する。
- **M4**: スモークテストで学習が正常に動作することを確認した後、本番100epoch学習を開始する。スモークテストのcheckpointをウォームスタートに使用することも検討できる。
- **M3-09**: batch=4でのK_s処理の問題が発覚した場合、M3-09に修正フィードバックを送ること。`K_s = min(cfg['training']['K_s'], batch_size // 2)` のような保護ロジックを追加する。
