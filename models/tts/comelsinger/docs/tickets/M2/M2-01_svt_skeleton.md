# M2-01: SVTModule クラス骨格

> **マイルストーン**: [M2: コアモジュール](../../13_milestones.md#m2-コアモジュール実装)
> **対応RQ**: RQ-02
> **依存チケット**: M1
> **ブロックするチケット**: M2-02
> **状態**: TODO

---

## 1. 目的とゴール

`svt_module.py` に `SVTModule(nn.Module)` の骨格を作成する。パラメータ `num_codebooks=12, codebook_size=1024, codebook_embed_dim=64, hidden_size=512, num_layers=4, num_heads=8, pitch_vocab_size=129, dropout=0.1` を `__init__` で受け取り、後続チケットが実装するサブモジュールのプレースホルダを配置する。

このチケットを完了すると M2-02〜M2-05 が独立して各サブモジュールを実装できる土台が整い、インスタンス化とパラメータ数検証（目標 ≈2M）が可能になる。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/svt_module.py
import torch
import torch.nn as nn

class SVTModule(nn.Module):
    """Singing Voice Transcription モジュール（論文 Section III-C）。
    音響トークン列から離散ピッチトークンを予測する。
    S2A 学習時は frozen (StopGrad) として使用する。
    """

    def __init__(
        self,
        num_codebooks: int = 12,
        codebook_size: int = 1024,
        codebook_embed_dim: int = 64,
        hidden_size: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        pitch_vocab_size: int = 129,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_embed_dim = codebook_embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pitch_vocab_size = pitch_vocab_size
        self.dropout = dropout
        # サブモジュールは M2-02〜M2-04 で実装
        # self.codebook_embs  → M2-02
        # self.input_proj     → M2-02
        # self.pos_enc        → M2-03
        # self.transformer    → M2-03
        # self.pitch_head     → M2-04

    def forward(self, acoustic_tokens, padding_mask=None):
        """M2-04 で実装。"""
        raise NotImplementedError("Implemented in M2-04")
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `svt_module.py` 骨格の作成・`__init__` 実装 |

## 4. 提供範囲とテスト項目

**含むもの**: `svt_module.py` 新規作成、`SVTModule.__init__` 実装、`forward` スタブ定義

**含まないもの**: サブモジュールの実際の実装（→ M2-02〜M2-04）、損失関数（→ M2-06）

### ユニットテスト

```bash
uv run python -c "
from models.tts.comelsinger.svt_module import SVTModule
m = SVTModule()
assert m.num_codebooks == 12
assert m.hidden_size == 512
assert m.pitch_vocab_size == 129
print('PASS: __init__ OK')
"
```

```bash
uv run python -c "
from models.tts.comelsinger.svt_module import SVTModule
import torch
m = SVTModule()
try:
    m.forward(torch.zeros(1,10,12).long())
    assert False
except NotImplementedError:
    print('PASS: forward stub OK')
"
```

### E2Eテスト

```bash
uv run python -c "
from models.tts.comelsinger.svt_module import SVTModule
m = SVTModule()
n = sum(p.numel() for p in m.parameters())
print(f'param count (skeleton): {n}')
assert n == 0, 'skeleton should have no params yet'
print('PASS: skeleton has no params (expected before M2-02~04)')
"
```

## 5. 懸念事項とレビュー項目

- **`__init__.py` の存在確認**: M0 で作成済みのはずだが、インポートパス `models.tts.comelsinger.svt_module` が通るか事前確認すること。
- **パラメータ数目標**: M2-04 完了後に ≈2M パラメータを目標とする。骨格段階では 0 で問題ない。

### レビュー項目

- [ ] `SVTModule()` がエラーなくインスタンス化できるか
- [ ] 全コンストラクタ引数がインスタンス属性として保持されているか
- [ ] `forward` が `NotImplementedError` を送出するか
- [ ] 型ヒントが `__init__` に付与されているか

## 6. フェーズ振り返り: 一から作り直すとしたら

**Transformer vs Conformer**: SVT に Conformer（畳み込み＋マルチヘッドアテンション複合ブロック）を使うとASR/SVT で実績があり、局所的韻律パターンの捕捉に有利。設計段階で `encoder_type: Literal["transformer", "conformer"]` を引数として持たせる。

**encoder-only vs decoder**: ピッチ予測はフレームアラインドであり、双方向文脈が有利。骨格段階から `bidirectional=True` フラグを持たせ、decoder への変更余地を残す。

**位置エンコーディング選択**: SinusoidalPosEmb（決定論的）vs RoPE（相対位置）vs ALiBi。骨格段階から `pos_encoding_type` 引数を公開しておくと実験が容易。

## 7. 後続タスクへの連絡事項

- **M2-02**: `codebook_embs` は `nn.ModuleList` として `self` に登録すること。`__init__` でのプレースホルダコメントを確認すること。
- **M2-03**: `SinusoidalPosEmb` は `llama_nar.py` に既存実装があるため、可能なら再利用すること。
- **M2-04**: `forward` の戻り値は `{"logits": Tensor, "probs": Tensor}` の辞書形式で固定する。
