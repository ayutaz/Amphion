# M3-04: SVT学習ループ実装

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07a
> **依存チケット**: M3-03
> **ブロックするチケット**: M3-05, M3-06
> **状態**: TODO

---

## 1. 目的とゴール

`tools/train_svt.py` に50Kステップの学習ループを追加する。各ステップで `forward → svt_loss → backward → grad_clip(1.0) → optimizer.step` の順に実行し、スケジューラをステップ毎に更新する。DataLoaderのイテレーションが尽きた場合は自動で再ループする（itertools.cycleパターン）。

## 2. 実装する内容の詳細

```python
# tools/train_svt.py (学習ループ部分)
import itertools
from torch.nn.utils import clip_grad_norm_

def train(model, dataloader, optimizer, scheduler, cfg):
    model.train()
    data_iter = itertools.cycle(dataloader)
    max_steps = cfg["training"]["max_steps"]

    for step in range(1, max_steps + 1):
        batch = next(data_iter)
        acoustic_tokens = batch["acoustic_tokens"].cuda()
        pitch_tokens = batch["pitch_tokens"].cuda()

        optimizer.zero_grad()
        loss_dict = model.compute_svt_loss(acoustic_tokens, pitch_tokens)
        loss = loss_dict["loss"]
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | 学習ループの実装・itertools.cycle設定・gradient clip確認 |

## 4. 提供範囲とテスト項目

**含むもの**: 50Kステップループ、forward/backward、grad_clip(1.0)、スケジューラ更新

**含まないもの**: ロギング（→ M3-05）、バリデーション（→ M3-06）、チェックポイント保存（→ M3-05）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.svt_module import SVTModule
# compute_svt_lossがlossキーを返すか確認
model = SVTModule()
acoustic = torch.zeros(2, 20, 8, dtype=torch.long)
pitch = torch.zeros(2, 20, dtype=torch.long)
out = model.compute_svt_loss(acoustic, pitch)
assert 'loss' in out and isinstance(out['loss'], torch.Tensor)
print('PASS: compute_svt_loss returns loss tensor')
"
```

```bash
uv run python -c "
import torch
from torch.nn.utils import clip_grad_norm_
from models.tts.comelsinger.svt_module import SVTModule
model = SVTModule()
# grad normがclip後に1.0以下になること
for p in model.parameters():
    p.grad = torch.ones_like(p) * 10.0
total_norm = clip_grad_norm_(model.parameters(), 1.0)
print(f'PASS: grad clipped (norm was {total_norm:.2f})')
"
```

### E2Eテスト

```bash
uv run python tools/train_svt.py \
    --config configs/comelsinger/svt_train.yaml \
    --max-steps 5 \
    --batch-size 2
# 5ステップ完了してexit 0になること
```

## 5. 懸念事項とレビュー項目

- **itertools.cycle**: 最終エポックで半端なバッチが発生する場合、drop_last=Trueとの組み合わせを確認する。
- **メモリリーク**: `.detach()` せずにloss履歴をlistに追記するとグラフが蓄積する。`loss.item()` で記録すること。

### レビュー項目

- [ ] grad_clip max_norm=1.0 が適用されていること
- [ ] scheduler.step() がoptimizer.step() の後であること
- [ ] optimizer.zero_grad() がステップ先頭で呼ばれていること

## 6. フェーズ振り返り: 一から作り直すとしたら

> 共通の設計判断（PyTorch Lightning vs Accelerate、実験管理、スケジューラ選択）は [M3_design_decisions.md](M3_design_decisions.md) を参照のこと。以下はこのチケット固有の設計判断を記載する。


- **PyTorch Lightning vs 素のAccelerate**: Lightningの `training_step` で実装すればloop管理が自動化され、clip_grad_normも `gradient_clip_val=1.0` で設定できる。
- **実験管理(W&B/MLflow)**: ループ内でのloss記録をW&Bで自動追跡する構成にするとハイパーパラメータ比較が容易。
- **GradNorm動的重み調整**: 固定のloss weightingではなくGradNormでλを動的調整する実装を最初から検討すべきだった。

## 7. 後続タスクへの連絡事項

- **M3-05**: `step` 変数と `loss_dict` は本ループから渡される。1000ステップ毎のロギング条件 `if step % 1000 == 0` を本ループ内に追記する形で統合すること。
- **M3-06**: `evaluate_svt()` はバリデーション用DataLoaderを別途受け取る設計とし、本学習ループからは引数として渡す。
- **M3-07**: 1000ステップの縮小実験で `max_steps=1000, batch_size=4` で動作することを確認してから本番スケールに移行する。
