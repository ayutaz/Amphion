# M1-10: compute_total_loss() 実装

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-04
> **依存チケット**: M1-07, M1-08, M1-09
> **ブロックするチケット**: M1-11
> **状態**: TODO

---

## 1. 目的とゴール

S2A 学習時の総損失 `compute_total_loss()` を実装する。SCL・FCL・SVT・マスク予測損失の4成分を論文規定の重みで統合し、各 λ を引数で上書き可能にする。

**ゴール**: `L_total = lambda_scl * L_SCL + lambda_fcl * L_FCL + lambda_svt * L_SVT + lambda_mask * L_mask` を計算し、スカラーテンソルと各成分の辞書を返す。

## 2. 実装する内容の詳細

```python
def compute_total_loss(
    l_scl: torch.Tensor,
    l_fcl: torch.Tensor,
    l_svt: torch.Tensor,
    l_mask: torch.Tensor,
    lambda_scl: float = 0.5,
    lambda_fcl: float = 1.0,
    lambda_svt: float = 0.1,
    lambda_mask: float = 0.3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    総損失関数。
    L_CL  = lambda_scl * L_SCL + lambda_fcl * L_FCL
    L_total = L_CL + lambda_svt * L_SVT + lambda_mask * L_mask
    デフォルト重みは論文 Table III 準拠。
    """
    l_cl = lambda_scl * l_scl + lambda_fcl * l_fcl
    total = l_cl + lambda_svt * l_svt + lambda_mask * l_mask
    return total, {
        "l_total": total,
        "l_cl": l_cl,
        "l_scl": l_scl,
        "l_fcl": l_fcl,
        "l_svt": l_svt,
        "l_mask": l_mask,
    }
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当 |
|---|---|---|
| 実装 | 1 | `losses.py` に `compute_total_loss` を実装 |
| 重み設計レビュー | 1 | 論文 Table III・CLAUDE.md のデフォルト重みとの照合 |
| テスト実装 | 1 | M1-11 のテスト先行作成（各 λ 上書きケース） |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲

`models/tts/comelsinger/losses.py` に `compute_total_loss` を追加

### 4.2 ユニットテスト

- デフォルト重みで `total == 0.5*l_scl + 1.0*l_fcl + 0.1*l_svt + 0.3*l_mask`（atol=1e-6）
- `lambda_svt=0.0` で `l_svt` の寄与がゼロになること
- 戻り値の dict に `"l_total"`, `"l_cl"`, `"l_scl"`, `"l_fcl"`, `"l_svt"`, `"l_mask"` の全キーが存在すること
- `total.requires_grad == True`（すべての入力が `requires_grad=True` のとき）

### 4.3 E2Eテスト

- `total.backward()` で各損失成分の勾配が到達すること
- `torch.manual_seed(42)` で再現性が保たれること

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- `L_CL = lambda_scl * L_SCL + lambda_fcl * L_FCL` という形式は CLAUDE.md と論文で重みが異なる可能性あり。論文 Table III を最終根拠として確認が必要
- `lambda_fcl=1.0` と `lambda_scl=0.5` の非対称な重みは FCL を支配的にする設計であり、学習初期の不安定化リスクがある

### 5.2 レビュー項目

- [ ] 論文記載の `λ_SCL=0.5, λ_FCL=1.0, λ_SVT=0.1, λ_mask=0.3` とデフォルト引数が一致しているか
- [ ] 戻り値の dict が TensorBoard ログ用途に使用可能なスカラーテンソルであること
- [ ] `l_cl` が独立したキーとして辞書に含まれているか（ログ用途）

## 6. フェーズ振り返り: 一から作り直すとしたら

- **InfoNCE vs NT-Xent**: SCL・FCL のどちらにも InfoNCE 変種を試し、損失スケールが揃うか検証すべきだった
- **数値安定性（BF16）**: 各成分を FP32 で計算して加算した後に `total` を返す設計にすると、AMP 混合精度でも安定する
- **損失関数クラス設計**: `LossWeights` のデータクラスを定義して重みをまとめると、設定ファイルからの読み込みや weight sweep が容易になる

## 7. 後続タスクへの連絡事項

- **M1-11**: `compute_total_loss` を含む `tests/test_losses.py` を作成。`torch.manual_seed(42)` での再現性テストを必須とする
- **M2 以降**: S2A `forward` メソッドから `compute_total_loss` を呼び出す際、`l_mask` は MaskGCT 既存の `forward_diffusion` 出力を直接渡す
- **学習パイプライン**: TensorBoard ログには戻り値 dict の全キーを `writer.add_scalar` で記録し、損失成分の推移を可視化すること
