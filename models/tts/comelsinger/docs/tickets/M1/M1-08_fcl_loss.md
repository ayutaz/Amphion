# M1-08: compute_fcl_loss() 実装

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-04
> **依存チケット**: なし
> **ブロックするチケット**: M1-10, M1-11
> **状態**: TODO

---

## 1. 目的とゴール

Frame-level Contrastive Loss（FCL）を実装する。フレームレベルの埋め込みペアに対し、ソフトラベル行列 `Y` を用いた対照学習を行い、参照音声からの prosody leakage をフレーム単位で防ぐ。

**ゴール**: `Y` が全ゼロのとき `loss == 0`。`tau=0.07` をデフォルトとし引数で変更可能。

## 2. 実装する内容の詳細

```python
def compute_fcl_loss(
    f_a: torch.Tensor,      # (B, L, D) — ソース側フレーム埋め込み
    f_b: torch.Tensor,      # (B, L, D) — 参照側フレーム埋め込み
    Y: torch.Tensor,        # (B, L, L) — ソフトラベル行列 {+1, 0}
    tau: float = 0.07,
) -> torch.Tensor:
    """
    ソフトラベル FCL（Frame-level Contrastive Loss）。
    Y が全ゼロのバッチ要素については loss=0 として安全にスキップ。
    """
    B, L, D = f_a.shape
    f_a = F.normalize(f_a, dim=-1)            # (B, L, D)
    f_b = F.normalize(f_b, dim=-1)            # (B, L, D)
    sim = torch.bmm(f_a, f_b.transpose(1, 2)) / tau  # (B, L, L)
    log_softmax = F.log_softmax(sim, dim=-1)  # (B, L, L)

    # Y の行和が 0 のフレームは損失対象外
    row_sum = Y.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, L, 1)
    soft_label = Y / row_sum                               # (B, L, L)

    # 有効フレームマスク: Y の行和 > 0
    valid_mask = (Y.sum(dim=-1) > 0).float()  # (B, L)

    per_frame_loss = -(soft_label * log_softmax).sum(dim=-1) * valid_mask  # (B, L)
    valid_count = valid_mask.sum().clamp(min=1.0)
    return per_frame_loss.sum() / valid_count
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当 |
|---|---|---|
| 実装 | 1 | `losses.py` に `compute_fcl_loss` を実装 |
| 数値安定性確認 | 1 | BF16/FP16 での NaN/Inf 検証、`tau=0.07` オーバーフロー対策 |
| レビュー | 1 | 論文 Section III-B との照合、ソフトラベル正規化の妥当性確認 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲

`models/tts/comelsinger/losses.py` に `compute_fcl_loss` を追加

### 4.2 ユニットテスト

- `Y` 全ゼロ `(B=2, L=10, D=64)` で `loss == 0.0`（exact）
- `f_a == f_b` かつ `Y` が単位行列のとき `loss ≈ 0`（atol=1e-4）
- ランダム入力・ランダム `Y` で `loss > 0` かつ有限値
- `tau=1.0` vs `tau=0.07` でスケール差が `loss` に反映されること
- FP16 入力で NaN/Inf なし（FP32 キャスト実装時）

### 4.3 E2Eテスト

- `compute_soft_label_matrix`（M1-04）の出力を `Y` として渡し、損失が逆伝播可能なこと
- `B=4, L=50, D=512` のリアルサイズで OOM なく動作すること

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- `sim / tau` が `tau=0.07` のとき値域が大きくなり、FP16 でオーバーフローの可能性あり。`f_a.float()` でキャストして計算する対策を検討
- `Y` の行和が全ゼロの場合に `valid_count=0` となる除算ゼロの防止が必要（`clamp(min=1.0)` で対処）
- ソフトラベルの正規化に `clamp(min=1e-8)` を使うことで、0除算を防ぐが微小誤差が生じる

### 5.2 レビュー項目

- [ ] `Y` のソフトラベル正規化（行和で除算）が論文数式(6)と一致しているか
- [ ] `valid_mask` の定義が論文のパディング除外と一致しているか
- [ ] `bmm` の次元が `(B, L, L)` となっているか（`f_a.transpose` と `f_b.transpose` を混同しないこと）

## 6. フェーズ振り返り: 一から作り直すとしたら

- **InfoNCE vs NT-Xent**: FCL は非対称（f_a→f_b 方向のみ）だが、双方向対称版を試してアブレーションする価値がある
- **数値安定性（BF16）**: `tau=0.07` の小温度は BF16 で exp がオーバーフローしやすい。最初から `log_sum_exp` のトリックを使う実装にすべきだった
- **損失関数クラス設計**: `tau` を学習可能パラメータにする場合は `nn.Module` にリファクタリングが必要。最初から `nn.Module` として設計するとその移行コストがゼロになる

## 7. 後続タスクへの連絡事項

- **M1-10**: `L_CL = 1.0 * L_SCL + 0.1 * L_FCL` として使用。FCL の重みは SCL の1/10
- **M1-11**: FP16 安定性テストで本関数を検証対象に含めること
- `Y` は M1-04（`compute_soft_label_matrix`）の出力を直接渡すことを前提としており、値域は `{+1, 0}`
