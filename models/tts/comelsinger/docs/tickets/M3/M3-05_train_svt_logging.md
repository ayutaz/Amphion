# M3-05: SVT学習ロギング・チェックポイント

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07a
> **依存チケット**: M3-04
> **ブロックするチケット**: M3-07
> **状態**: TODO

---

## 1. 目的とゴール

TensorBoardによるロギングを実装し、1000ステップ毎にval F1を記録してチェックポイントを保存する。バリデーション損失が最良値を更新した場合に `checkpoints/svt_best.pt` として保存する。ロギングとチェックポイント管理はM3-06のevaluate_svt()の結果を受け取って動作する。

## 2. 実装する内容の詳細

```python
# tools/train_svt.py (ロギング・チェックポイント部分)
from torch.utils.tensorboard import SummaryWriter
import torch

def setup_logging(log_dir="runs/svt"):
    return SummaryWriter(log_dir=log_dir)

def save_checkpoint(model, optimizer, step, val_f1, path):
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_f1": val_f1,
    }, path)

# 学習ループ内 (1000ステップ毎)
if step % 1000 == 0:
    val_f1 = evaluate_svt(model, val_loader)
    writer.add_scalar("train/loss", loss.item(), step)
    writer.add_scalar("val/f1", val_f1, step)
    save_checkpoint(model, optimizer, step, val_f1,
        f"checkpoints/svt_step{step}.pt")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        save_checkpoint(model, optimizer, step, val_f1,
            "checkpoints/svt_best.pt")
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | TensorBoard設定・チェックポイント保存・best model管理 |

## 4. 提供範囲とテスト項目

**含むもの**: TensorBoard SummaryWriter初期化、1000ステップ毎ロギング、svt_best.pt保存

**含まないもの**: F1計算ロジック（→ M3-06）、学習ループ本体（→ M3-04）

### ユニットテスト

```bash
uv run python -c "
import torch, os
from tools.train_svt import save_checkpoint
from models.tts.comelsinger.svt_module import SVTModule
import torch.optim as optim
model = SVTModule()
opt = optim.AdamW(model.parameters())
os.makedirs('checkpoints', exist_ok=True)
save_checkpoint(model, opt, 1000, 0.85, 'checkpoints/test_svt.pt')
ckpt = torch.load('checkpoints/test_svt.pt', map_location='cpu')
assert ckpt['step'] == 1000 and abs(ckpt['val_f1'] - 0.85) < 1e-6
print('PASS: checkpoint saved and loaded OK')
"
```

```bash
uv run python -c "
from torch.utils.tensorboard import SummaryWriter
import tempfile
with tempfile.TemporaryDirectory() as d:
    writer = SummaryWriter(log_dir=d)
    writer.add_scalar('test/loss', 1.0, 1)
    writer.close()
print('PASS: TensorBoard SummaryWriter OK')
"
```

### E2Eテスト

```bash
uv run python tools/train_svt.py \
    --config configs/comelsinger/svt_train.yaml \
    --max-steps 2000 --batch-size 2 --log-dir runs/test_svt
# runs/test_svtにeventファイルが生成され、checkpoints/svt_best.ptが存在すること
ls runs/test_svt/events.out.* && ls checkpoints/svt_best.pt
```

## 5. 懸念事項とレビュー項目

- **チェックポイントのディスク容量**: 1000ステップ毎に保存すると50ファイルになる。最新5件のみ保持するrotationを検討する。
- **val_f1の初期値**: `best_val_f1 = 0.0` として初期化し、最初のval評価で必ずsave_bestが発火するようにする。

### レビュー項目

- [ ] svt_best.pt が最良F1更新時のみ上書きされること
- [ ] TensorBoardのスカラーキーが "train/loss" と "val/f1" であること
- [ ] チェックポイントに step, val_f1 が含まれていること

## 6. フェーズ振り返り: 一から作り直すとしたら

- **PyTorch Lightning vs 素のAccelerate**: LightningはModelCheckpointコールバックがある。best model保存のロジックを自分で実装する必要がなくなる。
- **実験管理(W&B/MLflow)**: W&BはTensorBoardと互換APIがあり、`wandb.init(sync_tensorboard=True)` で移行コストが低い。最初からW&Bにする価値がある。
- **学習率スケジューラ選択**: CosineAnnealingLRのT_maxとmax_stepsの整合を自動で検証する仕組みが欲しかった。

## 7. 後続タスクへの連絡事項

- **M3-06**: `evaluate_svt()` の戻り値は `float`（F1スコア）の単一値とする。dictで複数指標を返す場合は本チケットのロギングコードも修正が必要。
- **M3-07**: スモークテスト（1000ステップ）では `--log-dir runs/smoke_svt` を指定してメインのrunsディレクトリと分ける。
- **M3-08**: SVTのbest checkpointパス（`checkpoints/svt_best.pt`）をS2A学習で読み込むため、パスを定数として `configs/comelsinger/s2a_train.yaml` にも記載する。
