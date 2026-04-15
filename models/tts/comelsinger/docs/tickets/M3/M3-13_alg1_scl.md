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

```python
# models/tts/comelsinger/contrastive_loss.py

import torch
import torch.nn.functional as F

def compute_scl_loss(g_a: torch.Tensor, g_b: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """シーケンスレベル対照損失（InfoNCE）を計算する。

    Args:
        g_a: (K_s, D) original サンプルのグローバル表現
        g_b: (K_s, D) perturbed サンプルのグローバル表現
        tau: 温度パラメータ (デフォルト 0.07)

    Returns:
        loss: スカラー損失値
    """
    g_a = F.normalize(g_a, dim=-1)
    g_b = F.normalize(g_b, dim=-1)
    logits = torch.matmul(g_a, g_b.T) / tau   # (K_s, K_s)
    labels = torch.arange(g_a.size(0), device=g_a.device)
    return F.cross_entropy(logits, labels)

def global_avg_pool(cond: torch.Tensor) -> torch.Tensor:
    """(B, T, D) -> (B, D) の平均プーリング。"""
    return cond.mean(dim=1)
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | compute_scl_loss()・global_avg_pool()実装・InfoNCE検証 |

## 4. 提供範囲とテスト項目

**含むもの**: `compute_scl_loss()`, `global_avg_pool()` （`contrastive_loss.py` に追加）

**含まないもの**: FCL（→ M3-14）、L_CL統合（→ M3-15）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.contrastive_loss import compute_scl_loss

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
from models.tts.comelsinger.contrastive_loss import global_avg_pool

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
from models.tts.comelsinger.contrastive_loss import compute_scl_loss, global_avg_pool

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

- **PyTorch Lightning vs 素のAccelerate**: フレームワーク非依存のモジュール。どちらでも同一実装。
- **実験管理(W&B/MLflow)**: SCL lossの推移とtau値をW&Bに記録。tau=0.07が最適かアブレーションで確認できる環境を最初から構築する。
- **GradNorm動的重み調整**: SCLとFCLの勾配スケールが大きく異なる場合、GradNormで自動調整することでL_CL全体の安定化が期待できる。

## 7. 後続タスクへの連絡事項

- **M3-14**: `compute_fcl_loss()` も同じ `contrastive_loss.py` に実装すること。SCLとFCLを同一ファイルにまとめることでimport管理を簡潔にする。
- **M3-15**: `L_CL = 1.0 * L_SCL + 0.1 * L_FCL` の重みは設定ファイル（lambda_scl=1.0, lambda_fcl=0.1）から読み込むこと。
- **M3-22**: tau=0.07はコードの定数ではなく設定ファイルから読み込む実装にすること（クロスチェック対象値）。
