# M2-04: SVTModule pitch_head + forward メソッド実装

> **マイルストーン**: [M2: コアモジュール](../../13_milestones.md#m2-コアモジュール実装)
> **対応RQ**: RQ-02
> **依存チケット**: M2-03
> **ブロックするチケット**: M2-05
> **状態**: TODO

---

## 1. 目的とゴール

`SVTModule` に `pitch_head`（`Linear(512, 129)`）を追加し、`forward` メソッドを完全実装する。出力は `{"logits": (B, L, 129), "probs": (B, L, 129)}` の辞書形式とする。

`probs` は `F.softmax(logits, dim=-1)` で算出し、全クラスの確率和が 1 になることを保証する。これが M2-06 の損失計算統合テストと M3 の SVT 学習パイプラインの入力となる。

## 2. 実装する内容の詳細

```python
import torch.nn.functional as F
from typing import Dict

# SVTModule.__init__ に追加
self.pitch_head = nn.Linear(hidden_size, pitch_vocab_size)

def forward(
    self,
    acoustic_tokens: torch.LongTensor,
    padding_mask: torch.BoolTensor | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Args:
        acoustic_tokens: (B, T, num_codebooks) 音響トークン
        padding_mask: (B, T) bool, True=パディング位置
    Returns:
        {"logits": (B,T,129), "probs": (B,T,129)}
    """
    x = self._embed(acoustic_tokens)          # (B, T, 512)
    x = self._encode(x, padding_mask)         # (B, T, 512)
    logits = self.pitch_head(x)               # (B, T, 129)
    probs  = F.softmax(logits, dim=-1)        # (B, T, 129)
    return {"logits": logits, "probs": probs}
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `pitch_head` 追加、`forward` 完全実装 |

## 4. 提供範囲とテスト項目

**含むもの**: `pitch_head` の `__init__` 登録、`forward` メソッドの完全実装

**含まないもの**: `freeze/unfreeze`（→ M2-05）、損失計算（→ M2-06）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.svt_module import SVTModule
m = SVTModule()
x = torch.randint(0, 1024, (2, 50, 12))
out = m(x)
assert out['logits'].shape == (2, 50, 129)
assert out['probs'].shape  == (2, 50, 129)
print('PASS: forward shape OK')
"
```

```bash
uv run python -c "
import torch
from models.tts.comelsinger.svt_module import SVTModule
m = SVTModule()
x = torch.randint(0, 1024, (2, 50, 12))
out = m(x)
assert torch.allclose(out['probs'].sum(-1), torch.ones(2, 50), atol=1e-5)
print('PASS: probs sum to 1 OK')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.svt_module import SVTModule
m = SVTModule()
n = sum(p.numel() for p in m.parameters())
print(f'total params: {n:,}')
# 目標 ≈2M: codebook_embs(786432) + input_proj(393728) + transformer(≈1M) + pitch_head(66048)
assert 1_500_000 < n < 3_000_000, f'param count out of expected range: {n}'
print('PASS: param count in expected range (~2M)')
"
```

## 5. 懸念事項とレビュー項目

- **`padding_mask` 位置の logits**: パディング位置の logits も計算される（損失計算時にマスクで除外）。`forward` 内でパディング位置を 0 埋めしないこと（損失側でマスクを適用する）。
- **数値安定性**: `F.softmax` の代わりに `F.log_softmax` + NLL loss を使う場合は `probs = logits.exp()` にする必要がある。損失計算方式を M2-06 と事前合意すること。

### レビュー項目

- [ ] `forward` の出力が `{"logits": ..., "probs": ...}` の辞書形式か
- [ ] `probs.sum(-1)` が `torch.ones(B, T)` に近いか（誤差 1e-5 以下）
- [ ] `padding_mask=None` で動作するか
- [ ] `pitch_head.out_features == 129` か

## 6. フェーズ振り返り: 一から作り直すとしたら

**Transformer vs Conformer**: `forward` の `_encode` 呼び出しは Conformer に差し替えても透過的。M2-03 の `_encode` を `nn.Module` としてカプセル化しておくと A/B テストが容易。

**encoder-only vs decoder**: `forward` が `padding_mask` を受け取る設計は encoder-only 前提。decoder に変更する場合は `tgt_mask`（因果マスク）も追加が必要。

**位置エンコーディング選択**: `_encode` 内で加算する設計のため、`forward` は位置エンコーディングに依存しない。RoPE へ変更する場合も `forward` の変更は不要。

## 7. 後続タスクへの連絡事項

- **M2-05**: `forward` が完成したため、`freeze()` 後に `forward` を呼び出しても勾配が計算されないことをテストすること。
- **M2-06**: 損失計算では `out["logits"]` を使用する（`F.cross_entropy` の入力は logits）。`out["probs"]` は推論時の可視化用途。
- **M3（SVT 学習）**: `acoustic_tokens` の shape は `(B, T, 12)` を前提とする。`collate_fn` が `(B, 12, T)` で返す場合は `.permute(0, 2, 1)` を `forward` 冒頭で行うこと。
