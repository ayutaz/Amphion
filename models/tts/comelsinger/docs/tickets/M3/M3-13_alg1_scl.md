# M3-13: Algorithm 1 シーケンスレベル対照学習（SCL）

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07b
> **依存チケット**: M3-12
> **ブロックするチケット**: M3-15
> **状態**: TODO

---

## 1. 目的とゴール

AvgPoolで `cond_B[:K_s]` と `cond_Bp[:K_s]` からグローバルベクトル `g_a` / `g_b`（shape: (K_s, 1024)）を生成し、`compute_scl_loss(tau=0.07)` でInfoNCEロスを計算する。SCLはシーケンス全体の音色表現がピッチ摂動に対してinvariantになることを学習させる。

## 2. 実装する内容の詳細

`compute_scl_loss` は M1-07 で `models/tts/comelsinger/losses.py` に実装済みのため、新規実装は行わない。学習ループからは `losses.py` の関数を直接インポートして使用すること。

```python
# tools/train_s2a.py での使用例（新規実装不要）

import torch
from models.tts.comelsinger.losses import compute_scl_loss

def global_avg_pool(cond: torch.Tensor) -> torch.Tensor:
    """(B, T, D) -> (B, D) の平均プーリング。学習ループ内のユーティリティ。"""
    return cond.mean(dim=1)

# AvgPool で g_a / g_b を生成して compute_scl_loss を呼び出す
g_a = global_avg_pool(cond_B[:K_s])   # (K_s, D)
g_b = global_avg_pool(cond_Bp[:K_s])  # (K_s, D)
l_scl = compute_scl_loss(g_a, g_b, tau=0.07)
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | compute_scl_loss()・global_avg_pool()実装・InfoNCE検証 |

## 4. 提供範囲とテスト項目

**含むもの**: `global_avg_pool()` ユーティリティ関数（学習ループ内）。`compute_scl_loss()` は M1-07 実装済みの `losses.py` から再利用（新規実装不要）

**含まないもの**: FCL（→ M3-14）、L_CL統合（→ M3-15）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.losses import compute_scl_loss

# 同一ベクトルのペアは損失が最小になること
g = torch.randn(8, 1024)
loss = compute_scl_loss(g, g.clone(), tau=0.07)
assert loss.item() < 0.1, f'Expected low loss for identical pairs, got {loss.item()}'
print(f'PASS: identical pairs loss={loss.item():.4f}')
"
```

```bash
uv run python -c "
import torch

def global_avg_pool(cond):
    return cond.mean(dim=1)

cond = torch.randn(8, 100, 1024)
g = global_avg_pool(cond)
assert g.shape == (8, 1024)
print('PASS: global_avg_pool shape OK')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.losses import compute_scl_loss

def global_avg_pool(cond):
    return cond.mean(dim=1)

cond_B_s = torch.randn(8, 100, 1024)
cond_Bp_s = torch.randn(8, 100, 1024)
g_a = global_avg_pool(cond_B_s)
g_b = global_avg_pool(cond_Bp_s)
loss = compute_scl_loss(g_a, g_b, tau=0.07)
assert not torch.isnan(loss) and not torch.isinf(loss)
print(f'PASS: SCL loss={loss.item():.4f} (finite)')
"
```

## 5. 懸念事項とレビュー項目

- **数値安定性**: tau=0.07は非常に小さく、logitsのスケールが大きくなりやすい。`F.normalize()` の前に勾配の爆発がないか確認する。
- **対称ロス**: InfoNCEは非対称（g_a → g_b方向のみ）。論文がsymmetric NCEを意図している場合、`(L(a→b) + L(b→a)) / 2` の実装が必要。

### レビュー項目

- [ ] `F.normalize()` が適用されていること
- [ ] `tau=0.07` が設定ファイルから読み込まれること
- [ ] 対角成分がpositive pairになっていること（labels = arange(K_s)）

## 6. フェーズ振り返り: 一から作り直すとしたら

> 共通の設計判断（PyTorch Lightning vs Accelerate、実験管理、スケジューラ選択）は [M3_design_decisions.md](M3_design_decisions.md) を参照のこと。以下はこのチケット固有の設計判断を記載する。


- **PyTorch Lightning vs 素のAccelerate**: フレームワーク非依存のモジュール。どちらでも同一実装。
- **実験管理(W&B/MLflow)**: SCL lossの推移とtau値をW&Bに記録。tau=0.07が最適かアブレーションで確認できる環境を最初から構築する。
- **GradNorm動的重み調整**: SCLとFCLの勾配スケールが大きく異なる場合、GradNormで自動調整することでL_CL全体の安定化が期待できる。

## 7. 後続タスクへの連絡事項

- **M3-14**: `compute_fcl_loss()` も M1-07/M1-08 で `losses.py` に実装済みの場合は再利用すること。未実装の場合は `losses.py` に追加してSCLと同一ファイルにまとめ、import管理を簡潔にする。
- **M3-15**: `L_CL = 1.0 * L_SCL + 0.1 * L_FCL` の重みは設定ファイル（lambda_scl=1.0, lambda_fcl=0.1）から読み込むこと。
- **M3-22**: tau=0.07はコードの定数ではなく設定ファイルから読み込む実装にすること（クロスチェック対象値）。
