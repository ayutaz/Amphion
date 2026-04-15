# M2-06: SVTModule 損失統合テスト

> **マイルストーン**: [M2: コアモジュール](../../13_milestones.md#m2-コアモジュール実装)
> **対応RQ**: RQ-02
> **依存チケット**: M2-05
> **ブロックするチケット**: M3-03
> **状態**: TODO

---

## 1. 目的とゴール

M1 で実装した `compute_svt_loss` と M2-01〜M2-05 で完成した `SVTModule` を組み合わせ、dummy バッチで `loss.backward()` が完走することを確認する。損失値が有限値（NaN/Inf なし）であることを保証する。

このチケットは実装ではなく統合テストが主目的であり、問題が見つかった場合は該当チケットへフィードバックする。

## 2. 実装する内容の詳細

```python
# tests/test_svt_integration.py として作成
import torch
import torch.nn.functional as F
from models.tts.comelsinger.svt_module import SVTModule

def test_svt_loss_backward():
    """dummy batchでcompute_svt_lossのbackwardが完走すること"""
    B, T = 2, 50
    model = SVTModule()
    model.train()

    acoustic_tokens = torch.randint(0, 1024, (B, T, 12))
    pitch_tokens    = torch.randint(0, 129,  (B, T))
    padding_mask    = torch.zeros(B, T, dtype=torch.bool)

    out    = model(acoustic_tokens, padding_mask)
    logits = out["logits"]          # (B, T, 129)

    # フラット化してCEを計算（padding位置を除外するためmask適用）
    valid = ~padding_mask           # (B, T) True=有効フレーム
    loss = F.cross_entropy(
        logits[valid],              # (N_valid, 129)
        pitch_tokens[valid],        # (N_valid,)
    )

    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"
    loss.backward()

    # 勾配確認
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"grad is None for {name}"
            assert torch.isfinite(p.grad).all(), f"grad has NaN/Inf: {name}"
    print(f"PASS: loss={loss.item():.4f}, backward OK")

if __name__ == "__main__":
    test_svt_loss_backward()
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| テストエージェント | 1 | 統合テストスクリプトの作成・実行・問題箇所のフィードバック |

## 4. 提供範囲とテスト項目

**含むもの**: 統合テストスクリプト `tests/test_svt_integration.py` の作成と実行

**含まないもの**: SVTModule 本体の変更（問題があれば M2-01〜M2-05 へフィードバック）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.svt_module import SVTModule
m = SVTModule()
out = m(torch.randint(0, 1024, (2, 50, 12)))
assert torch.isfinite(out['logits']).all(), 'logits contain NaN/Inf'
print('PASS: logits are finite')
"
```

```bash
uv run python -c "
import torch, torch.nn.functional as F
from models.tts.comelsinger.svt_module import SVTModule
m = SVTModule()
logits = m(torch.randint(0, 1024, (2, 10, 12)))['logits']
loss = F.cross_entropy(logits.reshape(-1, 129), torch.randint(0, 129, (20,)))
loss.backward()
assert torch.isfinite(loss), 'loss is not finite'
print('PASS: backward completes')
"
```

### E2Eテスト

```bash
uv run python tests/test_svt_integration.py
# 期待出力: PASS: loss=X.XXXX, backward OK
```

## 5. 懸念事項とレビュー項目

- **M1 の `compute_svt_loss` との連携**: M1 完了後に `compute_svt_loss` の引数と戻り値を確認し、このテストと整合させること。
- **損失の大きさ**: ランダム重みでの初期損失は `log(129) ≈ 4.86` 付近になるはず。極端に大きい場合（>10）は初期化に問題がある可能性がある。

### レビュー項目

- [ ] `loss.item()` が有限値か（NaN/Inf でないか）
- [ ] `loss.backward()` がエラーなく完走するか
- [ ] 全 `requires_grad=True` パラメータに勾配が計算されているか
- [ ] frozen 状態（`freeze()` 後）では勾配が計算されないか

## 6. フェーズ振り返り: 一から作り直すとしたら

**Transformer vs Conformer**: Conformer に変更しても統合テストのロジックは変わらない。テストが `SVTModule` の公開 API のみに依存しているため、実装変更に対して堅牢。

**encoder-only vs decoder**: decoder へ変更した場合、`padding_mask` の代わりに `tgt_mask` を使うため、テストのマスク生成部分を修正する必要がある。

**位置エンコーディング選択**: 統合テストは位置エンコーディングの実装に依存しない。`_encode` の内部実装を変更しても本テストは影響を受けない。

## 7. 後続タスクへの連絡事項

- **M3-03（SVT 損失計算モジュール）**: このテストで確認した `F.cross_entropy(logits[valid], pitch_tokens[valid])` のパターンを `compute_svt_loss` の実装に採用すること。
- **M3（SVT 学習パイプライン）**: 本テストを `pytest` に組み込み、CI で自動実行すること。
- **M2-11〜M2-15（S2A 側）**: SVT 統合テストが通過後、S2A 側から SVT を呼び出す統合テストを M2-14 で追加すること。
