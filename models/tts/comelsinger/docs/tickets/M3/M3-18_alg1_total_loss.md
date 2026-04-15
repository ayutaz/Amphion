# M3-18: Algorithm 1 総損失・backward・更新

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07b
> **依存チケット**: M3-15, M3-16, M3-17
> **ブロックするチケット**: M3-19
> **状態**: TODO

---

## 1. 目的とゴール

`L_total = 0.5 * L_CL + 0.5 * L_SVT + 0.3 * L_mask` を計算し、`backward → grad_clip(1.0) → optimizer.step → scheduler.step` を実行する関数を実装する。重み（lambda_cl, lambda_svt, lambda_mask）は設定ファイルから読み込む。Algorithm 1の最終ステップとしてパラメータ更新が完結する。

## 2. 実装する内容の詳細

```python
# tools/train_s2a.py の学習ループ内（総損失・更新部分）

from torch.nn.utils import clip_grad_norm_

def compute_and_update(
    l_cl: torch.Tensor,
    l_svt: torch.Tensor,
    l_mask: torch.Tensor,
    optimizer,
    scheduler,
    model,
    cfg: dict,
) -> dict:
    """総損失の計算・backward・パラメータ更新を行う。"""
    lw = cfg["loss_weights"]
    l_total = (lw["lambda_cl"] * l_cl
             + lw["lambda_svt"] * l_svt
             + lw["lambda_mask"] * l_mask)

    optimizer.zero_grad()
    l_total.backward()
    grad_norm = clip_grad_norm_(
        filter(lambda p: p.requires_grad, model.parameters()),
        max_norm=1.0)
    optimizer.step()
    scheduler.step()

    return {"l_total": l_total, "grad_norm": grad_norm,
            "l_cl": l_cl, "l_svt": l_svt, "l_mask": l_mask}
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | compute_and_update()実装・重み読込・backward/step確認 |

## 4. 提供範囲とテスト項目

**含むもの**: `compute_and_update()` 関数、総損失計算、backward、grad_clip、step

**含まないもの**: 各損失の計算（→ M3-15〜M3-17）、ロギング（→ M3-19）

### ユニットテスト

```bash
uv run python -c "
import torch
l_cl = torch.tensor(2.0, requires_grad=True)
l_svt = torch.tensor(1.5)  # no_grad経由のためrequires_grad=False
l_mask = torch.tensor(3.0, requires_grad=True)
# lambda値で正しく計算されること
l_total = 0.5 * l_cl + 0.5 * l_svt + 0.3 * l_mask
expected = 0.5*2.0 + 0.5*1.5 + 0.3*3.0  # = 2.65
assert abs(l_total.item() - expected) < 1e-5
print(f'PASS: L_total={l_total.item():.4f} (expected {expected})')
"
```

```bash
uv run python -c "
import torch, yaml
with open('configs/comelsinger/s2a_train.yaml') as f:
    cfg = yaml.safe_load(f)
lw = cfg['loss_weights']
# 設定ファイルの重みが正しいこと
assert lw['lambda_svt'] == 0.5
assert lw['lambda_mask'] == 0.3
print('PASS: loss weights from config OK')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch, yaml
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
from tools.train_s2a import compute_and_update

with open('configs/comelsinger/s2a_train.yaml') as f:
    cfg = yaml.safe_load(f)
model = CoMelSinger_S2A()
opt = torch.optim.AdamW(model.parameters())
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
l_cl = torch.tensor(1.0, requires_grad=True)
l_mask = torch.tensor(1.0, requires_grad=True)
l_svt = torch.tensor(1.0)
result = compute_and_update(l_cl, l_svt, l_mask, opt, sched, model, cfg)
assert 'l_total' in result and not torch.isnan(result['l_total'])
print(f\"PASS: L_total={result['l_total'].item():.4f}\")
"
```

## 5. 懸念事項とレビュー項目

- **lambda_cl の設定ファイル追加**: 現在の s2a_train.yaml には `lambda_cl` が明示されていない可能性がある。M3-01のファイルを確認し、必要なら追加する。lambda_cl=0.5を想定。
- **SVT損失のrequires_grad**: L_SVTが `torch.no_grad()` 下で計算されるため `requires_grad=False` になる可能性がある。L_totalのbackwardが成功するか確認する。

### レビュー項目

- [ ] L_total の重み係数が設定ファイルから読み込まれること
- [ ] grad_clip max_norm=1.0 が適用されること
- [ ] optimizer.step() の前に zero_grad() が呼ばれること

## 6. フェーズ振り返り: 一から作り直すとしたら

> 共通の設計判断（PyTorch Lightning vs Accelerate、実験管理、スケジューラ選択）は [M3_design_decisions.md](M3_design_decisions.md) を参照のこと。以下はこのチケット固有の設計判断を記載する。


- **PyTorch Lightning vs 素のAccelerate**: Lightningでは `optimizer_step` コールバックで grad_clip を一元管理できる。`gradient_clip_val=1.0` の1行で済む。
- **実験管理(W&B/MLflow)**: grad_normをW&Bに記録し、学習の安定性をリアルタイムモニタリングする仕組みを最初から組み込む。
- **GradNorm動的重み調整**: lambda_cl, lambda_svt, lambda_mask を固定値ではなくGradNormアルゴリズムで動的更新する実装を最初から設計すべきだった。特にL_SVTの重みがS2Aの学習フェーズによって最適値が変わる可能性がある。

## 7. 後続タスクへの連絡事項

- **M3-19**: `compute_and_update()` の戻り値 dict には `l_total`, `l_cl`, `l_svt`, `l_mask`, `grad_norm` が含まれる。TensorBoardへの記録はこの戻り値を使用すること。
- **M3-21**: スモークテスト（5epoch, batch=4）では全5損失のNaN/Infなしを確認する。`compute_and_update()` の戻り値でチェックする。
- **M3-22**: lambda_cl=0.5, lambda_svt=0.5, lambda_mask=0.3 がコード・設定・ドキュメントで一致するかをM3-22でクロスチェックする。
