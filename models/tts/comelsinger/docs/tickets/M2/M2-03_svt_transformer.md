# M2-03: SVTModule pos_enc + TransformerEncoder 実装

> **マイルストーン**: [M2: コアモジュール](../../13_milestones.md#m2-コアモジュール実装)
> **対応RQ**: RQ-02
> **依存チケット**: M2-02
> **ブロックするチケット**: M2-04
> **状態**: TODO

---

## 1. 目的とゴール

`SVTModule` に `SinusoidalPosEmb(512)` と `nn.TransformerEncoder`（4層、`norm_first=True`）を追加する。`src_key_padding_mask` によるパディングマスクに対応し、`(B, L, 512)` の埋め込みベクトルを文脈付きの `(B, L, 512)` 特徴量に変換する。

`norm_first=True`（Pre-LN）は学習安定性が高く、論文の設定に準拠する。

## 2. 実装する内容の詳細

```python
# svt_module.py SVTModule.__init__ に追加
from models.tts.maskgct.llama_nar import SinusoidalPosEmb  # 既存実装を再利用

self.pos_enc = SinusoidalPosEmb(hidden_size)  # (L, 512)

encoder_layer = nn.TransformerEncoderLayer(
    d_model=hidden_size,
    nhead=num_heads,
    dim_feedforward=hidden_size * 4,
    dropout=dropout,
    batch_first=True,
    norm_first=True,   # Pre-LN: 学習安定性向上
)
self.transformer = nn.TransformerEncoder(
    encoder_layer, num_layers=num_layers
)

# forward の一部（M2-04 で完成）
def _encode(
    self,
    x: torch.Tensor,
    padding_mask: torch.BoolTensor | None = None,
) -> torch.Tensor:
    """(B, L, 512) → (B, L, 512)"""
    L = x.size(1)
    pos = self.pos_enc(torch.arange(L, device=x.device))  # (L, 512)
    x = x + pos.unsqueeze(0)
    return self.transformer(x, src_key_padding_mask=padding_mask)
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `pos_enc`, `transformer`, `_encode` の実装 |

## 4. 提供範囲とテスト項目

**含むもの**: `pos_enc` / `transformer` の `__init__` 登録、`_encode` ヘルパーメソッド

**含まないもの**: `pitch_head`（→ M2-04）、`forward` 完全実装（→ M2-04）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.svt_module import SVTModule
m = SVTModule()
assert hasattr(m, 'pos_enc')
assert hasattr(m, 'transformer')
print('PASS: submodule registration OK')
"
```

```bash
uv run python -c "
import torch
from models.tts.comelsinger.svt_module import SVTModule
m = SVTModule()
x = torch.randn(2, 50, 512)
out = m._encode(x)
assert out.shape == (2, 50, 512), f'expected (2,50,512), got {out.shape}'
print('PASS: _encode shape OK')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.svt_module import SVTModule
m = SVTModule()
# padding_mask: True=パディング位置
x = torch.randn(2, 50, 512)
mask = torch.zeros(2, 50, dtype=torch.bool)
mask[1, 40:] = True   # バッチ1の後半10フレームはパディング
out = m._encode(x, padding_mask=mask)
assert out.shape == (2, 50, 512)
print('PASS: _encode with padding_mask OK')
"
```

## 5. 懸念事項とレビュー項目

- **SinusoidalPosEmb の API 確認**: `llama_nar.py` の `SinusoidalPosEmb` が `(L,)` のインデックスを受け取り `(L, dim)` を返すか確認すること。APIが異なる場合は独自実装を検討する。
- **`norm_first=True` の PyTorch バージョン**: PyTorch 1.11+ で利用可能。`requirements.txt` の制約を確認すること。

### レビュー項目

- [ ] `_encode` の出力 shape が `(B, L, 512)` か
- [ ] `padding_mask=None` でもエラーなく動作するか
- [ ] `padding_mask` 指定時にパディング位置の勾配が遮断されているか
- [ ] `norm_first=True` が TransformerEncoderLayer に設定されているか

## 6. フェーズ振り返り: 一から作り直すとしたら

**Transformer vs Conformer**: `nn.TransformerEncoderLayer` はシンプルだが、音声タスクでは Conformer（畳み込みモジュール追加）が有効。一から設計するなら `ConformerLayer` を実装し、A/B テストで比較する。

**encoder-only vs decoder**: SVT はフレームアラインドなピッチ予測タスクであり encoder-only が適切。ただし、因果マスクを加えた decoder 構成にすることでリアルタイム推論が可能になる。`causal` フラグを設計段階で用意する。

**位置エンコーディング選択**: SinusoidalPosEmb は固定だが、RoPE（Rotary Position Embedding）は相対位置を陰的に表現でき、長いシーケンスでの汎化性が高い。Transformer 本体の MHA に RoPE を組み込む設計も検討する。

## 7. 後続タスクへの連絡事項

- **M2-04**: `_encode` の出力 `(B, L, 512)` を `pitch_head` に入力する。`_encode` の return を `(B, L, 512)` で受け取ること。
- **M2-05**: `freeze()` は `transformer.parameters()` も含めて停止する。`pos_enc` は `nn.Module` でない場合は対象外となるため実装方法を確認すること。
- **M2-06**: `_encode` 中の `transformer` に勾配が流れることを単独学習時のテストで確認すること。
