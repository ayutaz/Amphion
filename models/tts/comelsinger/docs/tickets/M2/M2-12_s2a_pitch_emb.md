# M2-12: S2A pitch_emb 追加 + forward オーバーライド

> **マイルストーン**: [M2: コアモジュール](../../13_milestones.md#m2-コアモジュール実装)
> **対応RQ**: RQ-03
> **依存チケット**: M2-11
> **ブロックするチケット**: M2-13
> **状態**: TODO

---

## 1. 目的とゴール

`CoMelSinger_S2A` に `pitch_emb = nn.Embedding(129, 1024)` を追加し、`forward` をオーバーライドする。`cond = cond_emb(cond_code) + pitch_emb(pitch_tokens)` とすることでピッチ条件付けを実現する。`pitch_tokens=None` の場合は親クラスの挙動と同一（後方互換）を保証する。

## 2. 実装する内容の詳細

```python
# comelsinger_s2a.py CoMelSinger_S2A に追加
import torch
import torch.nn as nn

# __init__ に追加（super().__init__() 呼び出し後）
self.pitch_emb = nn.Embedding(pitch_vocab_size, 1024)

def forward(
    self,
    x0: torch.Tensor,
    x_mask: torch.Tensor,
    cond: torch.Tensor,
    cond_mask: torch.Tensor,
    pitch_tokens: torch.LongTensor | None = None,
    diffusion_step: torch.Tensor | None = None,
    **kwargs,
) -> dict:
    """pitch_tokens が None でない場合にピッチ埋め込みを cond に加算。
    pitch_tokens=None の場合は親クラスと同一の挙動。
    """
    if pitch_tokens is not None:
        # pitch_tokens: (B, T_cond) → pitch_emb: (B, T_cond, 1024)
        pitch_emb_out = self.pitch_emb(pitch_tokens)   # (B, T_cond, 1024)
        cond = cond + pitch_emb_out
    return super().forward(
        x0, x_mask, cond, cond_mask,
        diffusion_step=diffusion_step, **kwargs
    )
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `pitch_emb` 追加、`forward` オーバーライドの実装 |

## 4. 提供範囲とテスト項目

**含むもの**: `pitch_emb` モジュールの追加、`forward` オーバーライドの実装

**含まないもの**: `get_cond`（→ M2-13）、マスク損失統合（→ M2-14）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
model = CoMelSinger_S2A()
assert hasattr(model, 'pitch_emb')
assert model.pitch_emb.num_embeddings == 129
assert model.pitch_emb.embedding_dim  == 1024
print('PASS: pitch_emb registration OK')
"
```

```bash
uv run python -c "
import torch
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
model = CoMelSinger_S2A()
B, T = 2, 50
pitch_tokens = torch.randint(0, 129, (B, T))
out = model.pitch_emb(pitch_tokens)
assert out.shape == (B, T, 1024)
print('PASS: pitch_emb forward shape OK')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
model = CoMelSinger_S2A()
B, T = 1, 20
# pitch_tokens=None でも forward が動作するか（後方互換確認）
# 親クラスの forward に必要な引数は親クラスのシグネチャ確認後に補完
print('PASS: pitch_tokens=None backward compat placeholder OK')
print('NOTE: 親クラスのforward引数を確認後に実際のE2Eテストを実装すること')
"
```

## 5. 懸念事項とレビュー項目

- **`cond` の形状確認**: 親クラスの `cond_emb` が `(B, T_cond, 1024)` を返すかどうか `maskgct_s2a.py` で確認してから `pitch_emb` を加算すること。形状が異なる場合は線形変換を挟む。
- **`cond` と `pitch_tokens` の長さ**: `cond` と `pitch_tokens` の T_cond が一致している必要がある。不一致の場合はエラーメッセージを明確にすること。

### レビュー項目

- [ ] `pitch_emb.num_embeddings == 129` か
- [ ] `pitch_emb.embedding_dim == 1024` か
- [ ] `pitch_tokens=None` で親クラスと同一の挙動か
- [ ] `cond + pitch_emb_out` の形状が一致しているか

## 6. フェーズ振り返り: 一から作り直すとしたら

**継承 vs コンポジション**: `forward` オーバーライドは継承の典型的な使い方だが、親クラスの `forward` シグネチャ変更に弱い。コンポジションで `self.base_model.forward(modified_cond, ...)` を呼び出す設計なら親クラスの変更の影響を局所化できる。

**LoRA vs QLoRA vs フルFT**: `pitch_emb` は LoRA の `modules_to_save` に追加される（M2-15）。LoRA 適用前の `pitch_emb` パラメータは `requires_grad=True` のままにする必要がある。`freeze` を誤って呼ばないよう注意する。

**ハイパーパラメータ管理**: `embedding_dim=1024` をハードコードせず、親クラスの `hidden_size` を参照する設計（`self.pitch_emb = nn.Embedding(pitch_vocab_size, self.hidden_size)`）にすると汎用性が高い。

## 7. 後続タスクへの連絡事項

- **M2-13**: `get_cond(cond_code, pitch_tokens)` では `cond_emb(cond_code) + pitch_emb(pitch_tokens)` を計算して返す。`pitch_emb` が追加済みであることを前提とする。
- **M2-14**: `forward` オーバーライドが `super().forward()` を呼び出す設計のため、親クラスの `compute_loss` が正常に動作するか統合テストで確認すること。
- **M2-15**: LoRA 適用後、`pitch_emb.weight` が `modules_to_save` として学習可能なまま保持されているかを確認すること。
