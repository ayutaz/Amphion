# M3-19: S2A学習ロギング

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07b
> **依存チケット**: M3-18
> **ブロックするチケット**: M3-20
> **状態**: TODO

---

## 1. 目的とゴール

TensorBoardで5種の損失（L_SCL, L_FCL, L_CL, L_SVT, L_mask, L_total）をエポック毎に記録する。Accelerateの `main_process_only` コンテキストを使いDDP環境での重複書き込みを防ぐ。学習率・grad_normも記録対象に含める。

## 2. 実装する内容の詳細

```python
# tools/train_s2a.py のロギング部分

def log_metrics(writer, metrics: dict, epoch: int, accelerator):
    """TensorBoardに損失・grad_normを記録する（main processのみ）。"""
    if not accelerator.is_main_process:
        return
    loss_keys = ["l_scl", "l_fcl", "l_cl", "l_svt", "l_mask", "l_total"]
    for key in loss_keys:
        if key in metrics:
            writer.add_scalar(f"train/{key}", metrics[key], epoch)
    writer.add_scalar("train/grad_norm", metrics.get("grad_norm", 0.0), epoch)
    writer.add_scalar("train/lr",
        metrics.get("lr", 0.0), epoch)
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | log_metrics()実装・DDP対応・6種損失ログ設計 |

## 4. 提供範囲とテスト項目

**含むもの**: `log_metrics()` 関数、TensorBoardへの6損失+grad_norm+lr記録、DDP対応

**含まないもの**: チェックポイント保存（→ M3-20）、バリデーション（→ M3-20）

### ユニットテスト

```bash
uv run python -c "
from torch.utils.tensorboard import SummaryWriter
import tempfile
with tempfile.TemporaryDirectory() as d:
    writer = SummaryWriter(log_dir=d)
    metrics = {'l_scl': 1.0, 'l_fcl': 0.5, 'l_cl': 1.05, 'l_svt': 0.8,
               'l_mask': 2.0, 'l_total': 1.625, 'grad_norm': 0.9, 'lr': 1e-5}
    for k, v in metrics.items():
        writer.add_scalar(f'train/{k}', v, 1)
    writer.close()
print('PASS: TensorBoard logging 8 scalars OK')
"
```

```bash
uv run python -c "
# is_main_process=False の場合にログが書かれないことを確認
class MockAccelerator:
    is_main_process = False

metrics = {'l_total': 1.0}
acc = MockAccelerator()
# log_metricsがエラーなく早期リターンすること
if not acc.is_main_process:
    pass  # early return
print('PASS: non-main process skips logging')
"
```

### E2Eテスト

```bash
uv run python tools/train_s2a.py \
    --config configs/comelsinger/s2a_train.yaml \
    --max-epochs 2 --batch-size 2 --log-dir runs/test_s2a
# TensorBoardのイベントファイルに6種損失が記録されること
uv run python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('runs/test_s2a')
ea.Reload()
scalars = ea.Tags()['scalars']
required = ['train/l_scl', 'train/l_fcl', 'train/l_total']
assert all(k in scalars for k in required), f'Missing: {set(required)-set(scalars)}'
print('PASS: all required scalars in TensorBoard')
"
```

## 5. 懸念事項とレビュー項目

- **エポック毎 vs ステップ毎**: 100epochの場合、ステップ毎記録はデータが多くなりすぎる。エポック単位で集計したsmoothed lossを記録することを推奨する。
- **Accelerate環境でのwriter**: `accelerator.prepare()` にwriterを渡さない。writerはmain processのみで初期化する。

### レビュー項目

- [ ] 6種の損失キーが全てTensorBoardに記録されること
- [ ] DDP環境でmain processのみが書き込むこと
- [ ] grad_normと学習率も記録されること

## 6. フェーズ振り返り: 一から作り直すとしたら

> 共通の設計判断（PyTorch Lightning vs Accelerate、実験管理、スケジューラ選択）は [M3_design_decisions.md](M3_design_decisions.md) を参照のこと。以下はこのチケット固有の設計判断を記載する。


- **PyTorch Lightning vs 素のAccelerate**: Lightningは `self.log()` 1行で全プロセス集計・TensorBoard記録・プログレスバー表示が自動化される。手動の `log_metrics()` 関数が不要になる。
- **実験管理(W&B/MLflow)**: W&BはTensorBoardより視覚化が優れており、複数実験の比較が容易。最初からW&B `wandb.log()` を採用すべきだった。
- **学習率スケジューラ選択**: 逆平方根スケジューラは急激な減衰をするため、TensorBoardのlrグラフで可視化しておくと問題の早期発見ができる。

## 7. 後続タスクへの連絡事項

- **M3-20**: チェックポイント保存のタイミング（10epoch毎）と本ロギングを同一ループで管理する。`log_metrics()` の呼び出し後にチェックポイント保存条件を確認する設計。
- **M3-21**: スモークテスト（5epoch）でもTensorBoardの記録が正常に行われることを確認する。損失値がNaN/Infでないことをeventファイルから検証する。
- **M3-22**: ログのスカラーキー名（例: `train/l_scl`）がコードと設定で一貫していることを確認する。
