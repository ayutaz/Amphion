# M2-14: S2A マスク損失計算統合確認

> **マイルストーン**: [M2: コアモジュール](../../13_milestones.md#m2-コアモジュール実装)
> **対応RQ**: RQ-03
> **依存チケット**: M2-13
> **ブロックするチケット**: M2-15
> **状態**: TODO

---

## 1. 目的とゴール

`pitch_emb` 追加後も親クラスの `compute_loss → loss_t → forward_diffusion` が正常動作することを確認する。`L_mask.item()` が有限値であること、および `pitch_emb.weight.grad is not None` となることで、勾配が `pitch_emb` まで伝播していることを保証する。

このチケットは実装よりも統合テストが主目的だが、親クラスとの接続部分で修正が必要な場合はここで対応する。

## 2. 実装する内容の詳細

```python
# tests/test_s2a_mask_loss.py
import torch
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A

def test_mask_loss_backward():
    """pitch_emb追加後もcompute_lossが正常動作し勾配が伝播すること"""
    model = CoMelSinger_S2A()
    model.train()

    B, T_x, T_cond = 2, 40, 50
    # 親クラスの compute_loss に渡す引数（maskgct_s2a.py のシグネチャを要確認）
    x0     = torch.randint(0, 1024, (B, 8, T_x))   # (B, num_rvq, T)
    x_mask = torch.ones(B, T_x)
    # cond: get_cond を使って pitch 埋め込み込みの cond を作成
    cond_code    = torch.randint(0, 1024, (B, T_cond))
    pitch_tokens = torch.randint(0, 129,  (B, T_cond))
    cond = model.get_cond(cond_code, pitch_tokens)  # (B, T_cond, 1024)

    # 親クラスの compute_loss を呼び出す
    loss_dict = model.compute_loss(x0, x_mask, cond)
    L_mask = loss_dict["loss"] if isinstance(loss_dict, dict) else loss_dict
    assert torch.isfinite(L_mask), f"L_mask is not finite: {L_mask.item()}"
    L_mask.backward()
    assert model.pitch_emb.weight.grad is not None, "pitch_emb.weight.grad is None"
    assert torch.isfinite(model.pitch_emb.weight.grad).all(), "pitch_emb grad has NaN/Inf"
    print(f"PASS: L_mask={L_mask.item():.4f}, pitch_emb grad OK")

if __name__ == "__main__":
    test_mask_loss_backward()
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| テストエージェント | 1 | 統合テストスクリプトの作成・実行・問題フィードバック |

## 4. 提供範囲とテスト項目

**含むもの**: 統合テストスクリプト `tests/test_s2a_mask_loss.py` の作成と実行

**含まないもの**: `compute_loss` 本体の変更（親クラスに問題がない限り）、LoRA 適用（→ M2-15）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
model = CoMelSinger_S2A()
assert hasattr(model, 'compute_loss'), 'compute_loss must be inherited from parent'
print('PASS: compute_loss inherited OK')
"
```

```bash
uv run python -c "
import torch
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
model = CoMelSinger_S2A()
# pitch_emb が requires_grad=True であること
assert model.pitch_emb.weight.requires_grad, 'pitch_emb should require grad'
print('PASS: pitch_emb requires_grad OK')
"
```

### E2Eテスト

```bash
uv run python tests/test_s2a_mask_loss.py
# 期待出力: PASS: L_mask=X.XXXX, pitch_emb grad OK
```

## 5. 懸念事項とレビュー項目

- **`compute_loss` のシグネチャ**: 親クラスの `compute_loss` の引数が `(x0, x_mask, cond)` と異なる場合、テストスクリプトを修正すること。`maskgct_s2a.py` を事前確認すること。
- **勾配の計算グラフ**: `get_cond` で作成した `cond` テンソルが `compute_loss` の計算グラフに正しく組み込まれているか確認すること。`cond.requires_grad` が `True` であることを確認する。

### レビュー項目

- [ ] `L_mask.item()` が有限値か
- [ ] `pitch_emb.weight.grad is not None` か
- [ ] `pitch_emb.weight.grad` に NaN/Inf がないか
- [ ] `backward()` がエラーなく完走するか

## 6. フェーズ振り返り: 一から作り直すとしたら

**継承 vs コンポジション**: `super().compute_loss()` を呼び出す設計は継承の典型だが、親クラスの実装変更に弱い。コンポジション設計なら `self.base_model.compute_loss(...)` として依存を明示的に管理できる。

**LoRA vs QLoRA vs フルFT**: `pitch_emb.weight.grad` の確認はフルFT を前提としている。LoRA 適用後（M2-15）は `pitch_emb.weight` は `modules_to_save` として保持されるため、同様の確認が可能。

**ハイパーパラメータ管理**: 損失重み（`lambda_mask`）を `compute_loss` の中で適用する設計にするか、学習ループで適用するかを M2-14 の段階で決定する必要がある。

## 7. 後続タスクへの連絡事項

- **M2-15**: `pitch_emb.weight.grad` の確認はフルFT 時（M2-14）と LoRA 適用後（M2-15）で同様に実施すること。
- **M3（S2A 学習ループ）**: `compute_loss` の戻り値が辞書形式か単一テンソルかを本テストで確認し、学習ループの実装に反映すること。
- **M3（対照学習）**: 本テストは `L_mask` のみを確認する。`L_CL`（対照学習損失）の組み込みは M3 のコントラスティブ損失チケットで行う。
