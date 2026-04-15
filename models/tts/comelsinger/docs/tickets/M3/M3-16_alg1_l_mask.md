# M3-16: Algorithm 1 マスク予測損失（L_mask）

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07b
> **依存チケット**: M3-12
> **ブロックするチケット**: M3-18
> **状態**: TODO

---

## 1. 目的とゴール

`s2a_model.compute_loss()` を呼び出してマスク位置に対する `F.cross_entropy` ロス（L_mask）を計算する。MaskGCTのマスク予測損失をそのまま流用し、ピッチ条件付きforward（cond_B）の下で音響トークンのマスク予測精度を最適化する。マスクサンプリングはMaskGCT既存実装に委ねる。

## 2. 実装する内容の詳細

```python
# tools/train_s2a.py の学習ループ内（L_mask計算部分）

def compute_l_mask(
    s2a_model,
    acoustic_tokens: torch.Tensor,
    cond_B: torch.Tensor,
    mask_ratio: float = 0.5,
) -> torch.Tensor:
    """MaskGCTのマスク予測損失を計算する。

    Args:
        s2a_model: CoMelSinger_S2A モデル（LoRA適用済み）
        acoustic_tokens: (B, T, Q) 音響トークン (Q=8 RVQ codebooks)
        cond_B: (B, T, D) original ピッチ条件付き埋め込み
        mask_ratio: マスク率 (デフォルト0.5)

    Returns:
        l_mask: マスク予測損失（スカラー）
    """
    loss_dict = s2a_model.compute_loss(
        acoustic_tokens=acoustic_tokens,
        cond=cond_B,
        mask_ratio=mask_ratio,
    )
    return loss_dict["loss"]
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | compute_l_mask()実装・MaskGCT既存APIとの統合 |

## 4. 提供範囲とテスト項目

**含むもの**: `compute_l_mask()` 関数、`s2a_model.compute_loss()` との統合

**含まないもの**: L_mask内部のマスクサンプリング（→ MaskGCT既存実装）、L_total統合（→ M3-18）

### ユニットテスト

```bash
uv run python -c "
from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A
# compute_lossメソッドが存在することを確認
model = MaskGCT_S2A()
assert hasattr(model, 'compute_loss'), 'compute_loss method not found'
print('PASS: MaskGCT_S2A has compute_loss method')
"
```

```bash
uv run python -c "
import torch
# cross_entropyがマスク位置のみに適用されることを確認
logits = torch.randn(4, 20, 1024)
targets = torch.randint(0, 1024, (4, 20))
mask = torch.randint(0, 2, (4, 20)).bool()
loss = torch.nn.functional.cross_entropy(
    logits[mask], targets[mask])
assert not torch.isnan(loss)
print(f'PASS: masked cross_entropy loss={loss.item():.4f}')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
from tools.train_s2a import compute_l_mask

model = CoMelSinger_S2A()
acoustic = torch.zeros(4, 100, 8, dtype=torch.long)
cond_B = torch.randn(4, 100, 1024)
loss = compute_l_mask(model, acoustic, cond_B)
assert loss.item() > 0 and not torch.isnan(loss)
print(f'PASS: L_mask={loss.item():.4f}')
"
```

## 5. 懸念事項とレビュー項目

- **MaskGCT APIの互換性**: `compute_loss()` の引数が `cond_B` を受け取れるか MaskGCT_S2A のAPI確認が必要。条件付けの方法がピッチ埋め込み追加後も変わらないことを検証する。
- **マスク率の設定**: mask_ratio=0.5をデフォルトとするが、MaskGCT論文のスケジュールと整合させる必要がある。

### レビュー項目

- [ ] `s2a_model.compute_loss()` が `cond` 引数を受け取れること
- [ ] L_mask が正の値であること
- [ ] マスク位置のみにcross_entropyが適用されていること

## 6. フェーズ振り返り: 一から作り直すとしたら

- **PyTorch Lightning vs 素のAccelerate**: MaskGCTのcompute_lossをLightningのtraining_stepに組み込む場合、forward/lossの分離が自然になる。
- **実験管理(W&B/MLflow)**: L_maskのマスク率ごとの推移をW&Bで記録し、最適なマスクスケジュールを実験的に決定する。
- **Algorithm 1の関数分解粒度**: compute_l_mask を s2a_model のメソッドとして実装する方が、モデル内部のAPIとして自然。外部関数として実装するとモデルのAPI変更時に対応が困難。

## 7. 後続タスクへの連絡事項

- **M3-18**: `L_total = 0.5 * L_CL + 0.5 * L_SVT + 0.3 * L_mask` の `L_mask` として本関数の出力を使用する。lambda_mask=0.3は設定ファイルから読み込む。
- **M3-19**: L_mask単独をTensorBoardに記録すること。5種類の損失（L_SCL, L_FCL, L_CL, L_SVT, L_mask, L_total）の中の1つ。
- **M3-22**: lambda_mask=0.3 がコード・設定・ドキュメントで一致するかをM3-22でクロスチェックする。
