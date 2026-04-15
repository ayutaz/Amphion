# M1-07: compute_scl_loss() 実装

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-04
> **依存チケット**: なし
> **ブロックするチケット**: M1-10, M1-11
> **状態**: TODO

---

## 1. 目的とゴール

Sequence-level Contrastive Loss（SCL）を実装する。参照音声からの prosody leakage を防ぐため、シーケンスレベルの埋め込みを NT-Xent 対称版で対照学習する。

**ゴール**: `g_a == g_b` のとき `loss ≈ 0`。`tau=0.07` をデフォルトとし引数で変更可能。

## 2. 実装する内容の詳細

```python
def compute_scl_loss(g_a: torch.Tensor, g_b: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """NT-Xent 対称版。g_a, g_b: (K_s, D), K_s=8"""
    g_a = F.normalize(g_a, dim=-1)
    g_b = F.normalize(g_b, dim=-1)
    sim = torch.matmul(g_a, g_b.T) / tau  # (K_s, K_s)
    labels = torch.arange(g_a.size(0), device=g_a.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当 |
|---|---|---|
| 実装 | 1 | losses.py に関数実装 |
| レビュー | 1 | 論文 Section III-B との照合 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲

`models/tts/comelsinger/losses.py` に `compute_scl_loss` を実装

### 4.2 ユニットテスト

- `g_a == g_b` で loss ≈ 0（atol=1e-4）
- ランダム入力で loss > 0
- `tau=1.0`（大）vs `tau=0.07`（小）で小 tau ほど高 loss
- `K_s=1` で動作確認
- FP16 入力で NaN/Inf なし

### 4.3 E2Eテスト

- S2A forward パスで `L_SCL` が逆伝播可能なこと

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- `tau=0.07` は FP16 で `sim/tau` がオーバーフローする可能性。FP32 キャストを検討
- `K_s` の意味: バッチ内 SCL 用サンプル数（8）であり固定値ではない

### 5.2 レビュー項目

- [ ] `F.normalize` の dim が正しいか
- [ ] 双方向 CE の平均が正しく計算されているか
- [ ] `labels` のデバイスが入力と一致しているか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **InfoNCE vs NT-Xent**: InfoNCE（片方向）の方がシンプルで勾配解釈が容易。対称版との性能差をアブレーションで検証する
- **SupCon**: 同一話者サンプルを複数正例として扱える。話者数が少ない M4Singer で有効な可能性
- **クラス vs 関数**: 温度パラメータを学習可能にする場合は `nn.Module` クラス設計が適切

## 7. 後続タスクへの連絡事項

- **M1-10**: `L_CL = 1.0 * L_SCL + 0.1 * L_FCL` として使用。SCL の重みが FCL の10倍
- FP16 安定性問題は M1-11 の FP16 テストで早期検出
