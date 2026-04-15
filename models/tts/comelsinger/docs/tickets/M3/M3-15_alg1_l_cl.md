# M3-15: Algorithm 1 L_CL統合

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07b
> **依存チケット**: M3-13, M3-14
> **ブロックするチケット**: M3-18
> **状態**: TODO

---

## 1. 目的とゴール

`L_CL = lambda_scl * L_SCL + lambda_fcl * L_FCL`（lambda_scl=1.0, lambda_fcl=0.1）を計算する `compute_l_cl()` 関数を実装する。重みは設定ファイル `s2a_train.yaml` から読み込む。M3-13とM3-14の成果物を統合するシンプルな集約ステップ。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/contrastive_loss.py に追加

def compute_l_cl(
    l_scl: torch.Tensor,
    l_fcl: torch.Tensor,
    lambda_scl: float = 1.0,
    lambda_fcl: float = 0.1,
) -> torch.Tensor:
    """対照学習損失の重み付き和を計算する。

    Args:
        l_scl: シーケンスレベル対照損失（スカラー）
        l_fcl: フレームレベル対照損失（スカラー）
        lambda_scl: SCL重み（要件定義書12修正値: 1.0）
        lambda_fcl: FCL重み（要件定義書12修正値: 0.1）

    Returns:
        l_cl: 対照学習総損失（スカラー）
    """
    l_cl = lambda_scl * l_scl + lambda_fcl * l_fcl
    return l_cl
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | compute_l_cl()実装・重み値の検証 |

## 4. 提供範囲とテスト項目

**含むもの**: `compute_l_cl()` 関数（`contrastive_loss.py` に追加）

**含まないもの**: SCL計算（→ M3-13）、FCL計算（→ M3-14）、L_total統合（→ M3-18）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.contrastive_loss import compute_l_cl

l_scl = torch.tensor(2.0)
l_fcl = torch.tensor(3.0)
l_cl = compute_l_cl(l_scl, l_fcl, lambda_scl=1.0, lambda_fcl=0.1)
expected = 1.0 * 2.0 + 0.1 * 3.0  # = 2.3
assert abs(l_cl.item() - expected) < 1e-5, f'Expected {expected}, got {l_cl.item()}'
print(f'PASS: L_CL = {l_cl.item():.4f} (expected {expected})')
"
```

```bash
uv run python -c "
import yaml, torch
from models.tts.comelsinger.contrastive_loss import compute_l_cl

with open('configs/comelsinger/s2a_train.yaml') as f:
    cfg = yaml.safe_load(f)
lw = cfg['loss_weights']
l_cl = compute_l_cl(torch.tensor(1.0), torch.tensor(1.0),
    lambda_scl=lw['lambda_scl'], lambda_fcl=lw['lambda_fcl'])
expected = lw['lambda_scl'] + lw['lambda_fcl']
assert abs(l_cl.item() - expected) < 1e-5
print(f'PASS: L_CL from config = {l_cl.item():.4f}')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.contrastive_loss import compute_l_cl

# 勾配が伝播すること
l_scl = torch.tensor(2.0, requires_grad=True)
l_fcl = torch.tensor(3.0, requires_grad=True)
l_cl = compute_l_cl(l_scl, l_fcl)
l_cl.backward()
assert l_scl.grad is not None and l_fcl.grad is not None
assert abs(l_scl.grad.item() - 1.0) < 1e-5  # lambda_scl=1.0
assert abs(l_fcl.grad.item() - 0.1) < 1e-5  # lambda_fcl=0.1
print('PASS: gradient propagation OK')
"
```

## 5. 懸念事項とレビュー項目

- **重みの出典**: lambda_scl=1.0, lambda_fcl=0.1 は要件定義書12の修正値。論文原著値（λ_SCL=0.5, λ_FCL=1.0）と異なる。設定ファイルに明記する。
- **スケールの差異**: L_SCL と L_FCL のスケールが大きく異なる場合、lambda値のみで調整するのは限界がある。

### レビュー項目

- [ ] lambda_scl=1.0, lambda_fcl=0.1 が設定ファイルから読み込まれること
- [ ] 計算式が `1.0 * L_SCL + 0.1 * L_FCL` であること
- [ ] 勾配がL_SCLとL_FCLの両方に伝播すること

## 6. フェーズ振り返り: 一から作り直すとしたら

> 共通の設計判断（PyTorch Lightning vs Accelerate、実験管理、スケジューラ選択）は [M3_design_decisions.md](M3_design_decisions.md) を参照のこと。以下はこのチケット固有の設計判断を記載する。


- **PyTorch Lightning vs 素のAccelerate**: フレームワーク非依存の純粋な数式実装。どちらでも同一。
- **実験管理(W&B/MLflow)**: L_SCL, L_FCL, L_CL の3値を個別にW&Bに記録し、各損失の相対的な貢献をモニタリングする。
- **GradNorm動的重み調整**: lambda_scl と lambda_fcl を固定値ではなく学習中に動的調整するGradNorm実装を最初から検討すべきだった。実験でL_FCLが支配的になるケースで有効。

## 7. 後続タスクへの連絡事項

- **M3-18**: `L_total = 0.5 * L_CL + 0.5 * L_SVT + 0.3 * L_mask` の `L_CL` として本関数の出力を使用する。
- **M3-19**: L_SCL, L_FCL, L_CL の3値を個別にTensorBoardに記録すること。`l_cl` だけでなく `l_scl` と `l_fcl` も `compute_l_cl()` の内部でログ用に返せるよう戻り値の設計を検討する。
- **M3-22**: lambda_scl=1.0, lambda_fcl=0.1 がコード・設定・ドキュメントで一致するかをM3-22でクロスチェックする。
