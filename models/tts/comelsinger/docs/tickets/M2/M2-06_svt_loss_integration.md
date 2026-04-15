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
from models.tts.comelsinger.svt_module import SVTModule
from models.tts.comelsinger.losses import compute_svt_loss  # M1-09 実装の統合テスト

def test_svt_loss_backward():
    """dummy batchでcompute_svt_lossのbackwardが完走すること。
    手動の F.cross_entropy ではなく losses.py の compute_svt_loss を使う統合テスト。
    """
    B, T, N = 2, 50, 8  # N=ノート数
    model = SVTModule()
    model.train()

    acoustic_tokens = torch.randint(0, 1024, (B, T, 12))
    pitch_tokens    = torch.randint(0, 129,  (B, T))
    durations       = torch.rand(B, N) * 0.5 + 0.1    # GT デュレーション（秒）
    dur_preds       = torch.rand(B, N) * 0.5 + 0.1    # 予測デュレーション
    padding_mask    = torch.zeros(B, T, dtype=torch.bool)

    out    = model(acoustic_tokens, padding_mask)
    logits = out["logits"]          # (B, T, 129)

    # losses.py の compute_svt_loss を使って3成分損失を計算
    total, components = compute_svt_loss(
        logits=logits,
        targets=pitch_tokens,
        durations=durations,
        dur_preds=dur_preds,
        padding_mask=padding_mask,
    )

    assert torch.isfinite(total), f"total loss is not finite: {total.item()}"
    assert "l_ce" in components and "l_seg" in components and "l_dur" in components
    total.backward()

    # 勾配確認
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"grad is None for {name}"
            assert torch.isfinite(p.grad).all(), f"grad has NaN/Inf: {name}"
    print(f"PASS: total={total.item():.4f}, l_ce={components['l_ce'].item():.4f}, "
          f"l_seg={components['l_seg'].item():.4f}, l_dur={components['l_dur'].item():.4f}, backward OK")

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
import torch
from models.tts.comelsinger.svt_module import SVTModule
from models.tts.comelsinger.losses import compute_svt_loss
m = SVTModule()
out = m(torch.randint(0, 1024, (2, 10, 12)))
logits = out['logits']
total, comps = compute_svt_loss(
    logits=logits,
    targets=torch.randint(0, 129, (2, 10)),
    durations=torch.rand(2, 4) * 0.5 + 0.1,
    dur_preds=torch.rand(2, 4) * 0.5 + 0.1,
    padding_mask=torch.zeros(2, 10, dtype=torch.bool),
)
total.backward()
assert torch.isfinite(total), 'loss is not finite'
assert 'l_ce' in comps and 'l_seg' in comps and 'l_dur' in comps
print('PASS: compute_svt_loss backward completes, all 3 components present')
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

- **M3-03（SVT 損失計算モジュール）**: このテストは `losses.compute_svt_loss` を使った統合テストである。M1-09 実装の `compute_svt_loss(logits, targets, durations, dur_preds, padding_mask)` の引数シグネチャと戻り値 `(total, {"l_ce", "l_seg", "l_dur"})` に本テストを合わせてあること。
- **M3（SVT 学習パイプライン）**: 本テストを `pytest` に組み込み、CI で自動実行すること。
- **M2-11〜M2-15（S2A 側）**: SVT 統合テストが通過後、S2A 側から SVT を呼び出す統合テストを M2-14 で追加すること。
