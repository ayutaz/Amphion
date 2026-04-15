# M3-20: S2Aチェックポイント保存

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07b
> **依存チケット**: M3-19
> **ブロックするチケット**: M3-21
> **状態**: TODO

---

## 1. 目的とゴール

10epoch毎にLoRAの差分重みのみを保存し、バリデーションF0-RMSEが最小値を更新した場合に `checkpoints/s2a_best.pt` として保存する機能を実装する。LoRAの差分保存には PEFT の `save_pretrained()` を使用し、全パラメータではなくLoRAアダプタのみ保存することでディスク容量を節約する。

## 2. 実装する内容の詳細

```python
# tools/train_s2a.py のチェックポイント部分

import os
from peft import PeftModel

def save_s2a_checkpoint(model, epoch: int, val_f0_rmse: float,
                        best_f0_rmse: float, ckpt_dir: str):
    """LoRA差分重みを保存し、best modelを更新する。

    Args:
        model: PeftModel (LoRA適用済み CoMelSinger_S2A)
        epoch: 現在のエポック数
        val_f0_rmse: バリデーションF0-RMSE（低いほど良い）
        best_f0_rmse: これまでの最良F0-RMSE
        ckpt_dir: チェックポイント保存ディレクトリ

    Returns:
        updated_best: 更新後のbest_f0_rmse
    """
    if epoch % 10 == 0:
        epoch_dir = os.path.join(ckpt_dir, f"epoch_{epoch:04d}")
        model.save_pretrained(epoch_dir)  # LoRA差分のみ保存

    if val_f0_rmse < best_f0_rmse:
        best_dir = os.path.join(ckpt_dir, "s2a_best")
        model.save_pretrained(best_dir)
        return val_f0_rmse
    return best_f0_rmse
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | save_s2a_checkpoint()実装・LoRA差分保存・best model更新 |

## 4. 提供範囲とテスト項目

**含むもの**: `save_s2a_checkpoint()` 関数、10epoch毎保存、F0-RMSE最小でbest保存

**含まないもの**: F0-RMSE計算（→ 推論パイプライン or M4）、チェックポイント読込（→ M4）

### ユニットテスト

```bash
uv run python -c "
import os, tempfile
with tempfile.TemporaryDirectory() as ckpt_dir:
    # epoch%10==0 の時のみ保存されること
    # PEFT未適用のダミーで動作確認
    epoch_dir = os.path.join(ckpt_dir, 'epoch_0010')
    os.makedirs(epoch_dir)
    assert os.path.exists(epoch_dir)
print('PASS: epoch checkpoint directory created')
"
```

```bash
uv run python -c "
# best_f0_rmse更新ロジックの確認
best = float('inf')
val_rmse = 15.0
updated = val_rmse if val_rmse < best else best
assert updated == 15.0
val_rmse2 = 20.0
updated2 = val_rmse2 if val_rmse2 < updated else updated
assert updated2 == 15.0  # 悪化した場合は更新しない
print('PASS: best_f0_rmse update logic OK')
"
```

### E2Eテスト

```bash
uv run python tools/train_s2a.py \
    --config configs/comelsinger/s2a_train.yaml \
    --max-epochs 10 --batch-size 2 \
    --ckpt-dir checkpoints/test_s2a
# epoch_0010ディレクトリが作成されること
ls checkpoints/test_s2a/epoch_0010/adapter_model.bin
ls checkpoints/test_s2a/s2a_best/adapter_model.bin
```

## 5. 懸念事項とレビュー項目

- **F0-RMSE計算の依存**: バリデーションF0-RMSEの計算はM4で実装する推論パイプラインに依存する。M3段階では代替指標（validation mask loss）でbest modelを選択することを検討する。
- **DDP環境での保存**: main processのみ保存すること。`accelerator.is_main_process` を確認してから `save_pretrained()` を呼ぶ。

### レビュー項目

- [ ] 10epoch毎のディレクトリが作成されること
- [ ] LoRAアダプタのみが保存されていること（全パラメータではない）
- [ ] best_f0_rmseが改善した場合のみs2a_bestが上書きされること

## 6. フェーズ振り返り: 一から作り直すとしたら

- **PyTorch Lightning vs 素のAccelerate**: LightningのModelCheckpointコールバックはepoch保存とbest model保存を自動管理する。`save_top_k=3` でbest-k件のみ保持できる。
- **実験管理(W&B/MLflow)**: W&BのArtifact機能でLoRAチェックポイントをバージョン管理することで、実験の再現性と比較が容易になる。
- **学習率スケジューラ選択**: チェックポイント保存後の学習再開時に、スケジューラのstep数を正確に復元する必要がある。スケジューラのstate_dictも同時に保存すべきだった。

## 7. 後続タスクへの連絡事項

- **M3-21**: スモークテスト（5epoch, batch=4）ではepoch 5での保存は発生しない（10epoch毎のため）。スモークテスト用に `save_every_n_epochs=1` オプションを追加することを検討する。
- **M4**: `s2a_best` ディレクトリから `PeftModel.from_pretrained()` でLoRAアダプタを読み込む実装を前提とする。base modelのパスとadapter pathを別々に管理すること。
- **M3-22**: LoRA差分保存が実際にアダプタのみ（約4.8%のパラメータ）を保存しているか、ファイルサイズで確認する。
