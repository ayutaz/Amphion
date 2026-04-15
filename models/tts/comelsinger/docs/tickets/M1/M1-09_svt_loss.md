# M1-09: compute_svt_loss() 実装

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-04
> **依存チケット**: なし
> **ブロックするチケット**: M1-10, M1-11
> **状態**: TODO

---

## 1. 目的とゴール

SVT（Singing Voice Transcription）モジュールの3成分損失を実装する。クロスエントロピー損失・セグメント境界損失・ソフトデュレーション損失を組み合わせ、SVT モジュール（frozen）によるピッチ監督として機能させる。

**ゴール**: 戻り値 `(total, {"l_ce": ..., "l_seg": ..., "l_dur": ...})`。パディングを除外した CE、境界遷移ペナルティ、デュレーション分布一致を正しく計算する。

## 2. 実装する内容の詳細

```python
def compute_svt_loss(
    logits: torch.Tensor,       # (B, T, V) — SVT の出力ロジット
    targets: torch.Tensor,      # (B, T) long — ピッチトークン (0=無声)
    durations: torch.Tensor,    # (B, N) float — ノートデュレーション（秒）
    dur_preds: torch.Tensor,    # (B, N) float — 予測デュレーション
    padding_mask: torch.Tensor, # (B, T) bool — True=パディング位置
    lambda_seg: float = 3.0,
    lambda_dur: float = 5.0,
    delta: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    SVT 3成分損失。
    L_SVT = L_CE + lambda_seg * L_seg + lambda_dur * L_dur
    """
    # L_CE: パディング除外クロスエントロピー
    l_ce = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-1,
        reduction="none",
    )
    valid_mask = ~padding_mask.view(-1)
    l_ce = l_ce[valid_mask].mean()

    # L_seg: 境界遷移損失（隣接フレーム間のトークン変化をペナルティ）
    pred = logits.argmax(dim=-1)        # (B, T)
    boundary = (pred[:, 1:] != pred[:, :-1]).float()  # (B, T-1)
    target_boundary = (targets[:, 1:] != targets[:, :-1]).float()
    l_seg = F.binary_cross_entropy(boundary.clamp(1e-6, 1 - 1e-6), target_boundary)

    # L_dur: ソフトデュレーション損失（Huber 損失で δ=0.5）
    l_dur = F.huber_loss(dur_preds, durations, delta=delta)

    total = l_ce + lambda_seg * l_seg + lambda_dur * l_dur
    return total, {"l_ce": l_ce, "l_seg": l_seg, "l_dur": l_dur}
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当 |
|---|---|---|
| 実装 | 1 | `losses.py` に `compute_svt_loss` を実装 |
| 損失設計レビュー | 1 | 論文 Section III-C・数式(8)との照合、境界損失の定義確認 |
| テスト実装 | 1 | M1-11 のテスト先行作成（受入基準との対応付け） |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲

`models/tts/comelsinger/losses.py` に `compute_svt_loss` を追加

### 4.2 ユニットテスト

- 全パディングのバッチで `l_ce` が NaN でないこと（空テンソルの平均対策）
- `logits` がターゲットに完全一致するとき `l_ce ≈ 0`（atol=1e-4）
- `lambda_seg=0, lambda_dur=0` で `total == l_ce`
- `dur_preds == durations` のとき `l_dur == 0`
- 戻り値の dict に `"l_ce"`, `"l_seg"`, `"l_dur"` キーが含まれること

### 4.3 E2Eテスト

- `total` が逆伝播可能（`.backward()` でエラーなし）
- SVT モジュール（frozen）経由で勾配が `logits` に流れないこと（`stop_grad` 確認）

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- 全フレームがパディングの場合、`l_ce` の計算対象が空テンソルになり `mean()` が NaN になる。`valid_mask.any()` でガードが必要
- `L_seg` の `binary_cross_entropy` は `boundary` が連続値でない場合に定義が不明確。`F.binary_cross_entropy_with_logits` への変更を検討
- デュレーション `durations` の単位（秒 vs フレーム数）を学習パイプラインと統一する必要がある

### 5.2 レビュー項目

- [ ] `l_ce` の `ignore_index=-1` とパディング処理が一致しているか（`padding_mask` → `-1` への変換）
- [ ] `L_seg` の定義が論文数式(8)と一致しているか
- [ ] `delta=0.5` の Huber 損失が論文に記載の「soft duration loss」の意図と合致しているか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **損失関数クラス設計**: 3成分をそれぞれ `nn.Module` として分離すると、各損失の `lambda` を学習可能パラメータに昇格させやすい。最初からクラス設計にすべきだった
- **property-based testing**: `hypothesis` でランダムな `logits`・`targets`・`padding_mask` を生成し、「total が有限値かつ非負」を形式検証する
- **数値安定性（BF16）**: `F.cross_entropy` は BF16 で精度低下する。訓練時は常に FP32 で計算し、AMP の `autocast` スコープ外に置く設計が堅牢

## 7. 後続タスクへの連絡事項

- **M1-10**: `L_SVT = compute_svt_loss(...)` の戻り値 `total` を `lambda_SVT * L_SVT` として加算する
- **M1-11**: `padding_mask` が全 True（全パディング）の境界ケーステストを必ず含めること
- **SVT モジュール実装（M2以降）**: SVT の出力 `logits` と `dur_preds` をこの関数に渡す際のテンソル形状を合わせること
