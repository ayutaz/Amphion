# M3-14: Algorithm 1 フレームレベル対照学習（FCL）

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07b
> **依存チケット**: M3-12
> **ブロックするチケット**: M3-15
> **状態**: TODO

---

## 1. 目的とゴール

`f_a = cond_B[K_s:]`、`f_b = cond_Bp[K_s:]`（各: (N_f, T, 1024)）を取得し、soft label matrix Y（ピッチの近さに基づく重みラベル）を生成して `compute_fcl_loss()` でフレームレベル対照損失を計算する。soft labelにより、ピッチが近いフレームはポジティブペアとして扱う。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/contrastive_loss.py に追加

def build_soft_label_matrix(
    pitch_tokens: torch.Tensor, sigma: float = 2.0
) -> torch.Tensor:
    """ピッチトークンの近さに基づくsoft label行列を構築する。

    Args:
        pitch_tokens: (B, T) ピッチトークン
        sigma: ガウスカーネルの幅（セミトーン単位）

    Returns:
        Y: (B, T, T) soft label行列（行方向にsoftmax正規化済み）
    """
    B, T = pitch_tokens.shape
    p = pitch_tokens.float().unsqueeze(-1)   # (B, T, 1)
    diff = (p - p.transpose(1, 2)).abs()     # (B, T, T)
    Y = torch.exp(-diff ** 2 / (2 * sigma ** 2))
    Y = Y / Y.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return Y

def compute_fcl_loss(
    f_a: torch.Tensor, f_b: torch.Tensor,
    Y: torch.Tensor, tau: float = 0.07
) -> torch.Tensor:
    """フレームレベル対照損失を計算する。"""
    f_a = F.normalize(f_a, dim=-1)   # (B, T, D)
    f_b = F.normalize(f_b, dim=-1)
    sim = torch.bmm(f_a, f_b.transpose(1, 2)) / tau  # (B, T, T)
    loss = -(Y * F.log_softmax(sim, dim=-1)).sum(dim=-1).mean()
    return loss
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | build_soft_label_matrix()・compute_fcl_loss()実装・soft label検証 |

## 4. 提供範囲とテスト項目

**含むもの**: `build_soft_label_matrix()`, `compute_fcl_loss()` （`contrastive_loss.py` に追加）

**含まないもの**: SCL（→ M3-13）、L_CL統合（→ M3-15）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.contrastive_loss import build_soft_label_matrix

pitch = torch.zeros(4, 10, dtype=torch.long)  # 全フレームが同一ピッチ
Y = build_soft_label_matrix(pitch, sigma=2.0)
assert Y.shape == (4, 10, 10)
# 全フレーム同一ピッチなら均一分布になること
assert torch.allclose(Y[0, 0], torch.full((10,), 0.1), atol=1e-5)
print('PASS: soft label matrix uniform for same pitch')
"
```

```bash
uv run python -c "
import torch
from models.tts.comelsinger.contrastive_loss import compute_fcl_loss

B, T, D = 4, 20, 1024
f_a = torch.randn(B, T, D)
f_b = torch.randn(B, T, D)
Y = torch.eye(T).unsqueeze(0).expand(B, -1, -1)  # hard label (対角が1)
loss = compute_fcl_loss(f_a, f_b, Y)
assert not torch.isnan(loss) and not torch.isinf(loss)
print(f'PASS: FCL loss={loss.item():.4f}')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.contrastive_loss import build_soft_label_matrix, compute_fcl_loss

pitch = torch.randint(60, 80, (24, 50))  # FCL用サンプル数=24
f_a = torch.randn(24, 50, 1024)
f_b = torch.randn(24, 50, 1024)
Y = build_soft_label_matrix(pitch)
loss = compute_fcl_loss(f_a, f_b, Y)
assert loss.item() > 0
print(f'PASS: FCL E2E loss={loss.item():.4f}')
"
```

## 5. 懸念事項とレビュー項目

- **メモリ消費**: (B, T, T) の行列は T=200フレームで B=24の場合 24×200×200×4bytes≈38MB。GPU OOMに注意する。
- **パディングフレームの扱い**: Y行列のパディング行/列はゼロにして損失計算から除外する必要がある。

### レビュー項目

- [ ] soft label Yの各行の和が1.0であること（確率分布）
- [ ] 同一ピッチのフレームはY値が高いこと
- [ ] 損失がNaN/Infでないこと

## 6. フェーズ振り返り: 一から作り直すとしたら

> 共通の設計判断（PyTorch Lightning vs Accelerate、実験管理、スケジューラ選択）は [M3_design_decisions.md](M3_design_decisions.md) を参照のこと。以下はこのチケット固有の設計判断を記載する。


- **PyTorch Lightning vs 素のAccelerate**: フレームワーク非依存のモジュール。
- **実験管理(W&B/MLflow)**: soft label matrixをヒートマップとしてW&Bに記録することで、ピッチ類似度の視覚化が可能になる。
- **GradNorm動的重み調整**: FCLの勾配がSCLより大きくなりがちなため、GradNormによる自動重み調整で安定化を図るべきだった。

## 7. 後続タスクへの連絡事項

- **M3-15**: `compute_fcl_loss()` の戻り値はスカラーであり、`L_CL = 1.0 * L_SCL + 0.1 * L_FCL` に代入する。lambda_fcl=0.1は設定ファイルから読み込む。
- **M3-22**: sigma=2.0（soft labelのガウス幅）はコードのハードコードでなく設定ファイルに記載することを推奨する。クロスチェック対象外だが、実験再現性のため記録する。
- **M3-09**: FCLで使用する `f_a = cond_B[K_s:]` は split_batch() で取得した `s_a_f`（インデックス[8:]）に対応する。バッチ分割の順序と一致していることを確認すること。
