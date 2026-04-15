# M2-02: SVTModule codebook_embs + input_proj 実装

> **マイルストーン**: [M2: コアモジュール](../../13_milestones.md#m2-コアモジュール実装)
> **対応RQ**: RQ-02
> **依存チケット**: M2-01
> **ブロックするチケット**: M2-03
> **状態**: TODO

---

## 1. 目的とゴール

`SVTModule` に `codebook_embs`（`nn.ModuleList` 12×`Embedding(1024, 64)`）と `input_proj`（`Linear(768, 512)`）を追加する。`(B, L, 12)` の音響トークンを埋め込み後に連結し `(B, L, 512)` へ射影する `_embed` ヘルパーを実装する。

この変換がピッチ予測の入力特徴量の基盤となる。12 コードブックを独立した Embedding で表現することで、各コードブックの意味情報を個別に学習できる。

## 2. 実装する内容の詳細

```python
# svt_module.py SVTModule.__init__ に追加
self.codebook_embs = nn.ModuleList([
    nn.Embedding(codebook_size, codebook_embed_dim)
    for _ in range(num_codebooks)
])
# 12 * 64 = 768 → hidden_size=512
self.input_proj = nn.Linear(
    num_codebooks * codebook_embed_dim, hidden_size
)

def _embed(self, acoustic_tokens: torch.LongTensor) -> torch.Tensor:
    """(B, L, num_codebooks) → (B, L, hidden_size)"""
    # acoustic_tokens: (B, L, 12)
    embs = [
        emb(acoustic_tokens[..., i])   # (B, L, 64)
        for i, emb in enumerate(self.codebook_embs)
    ]
    x = torch.cat(embs, dim=-1)        # (B, L, 768)
    return self.input_proj(x)          # (B, L, 512)
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `codebook_embs`, `input_proj`, `_embed` の実装 |

## 4. 提供範囲とテスト項目

**含むもの**: `codebook_embs` / `input_proj` の `__init__` 登録、`_embed` ヘルパーメソッド

**含まないもの**: Transformer 本体（→ M2-03）、`forward` 完全実装（→ M2-04）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.svt_module import SVTModule
m = SVTModule()
assert len(m.codebook_embs) == 12
assert m.codebook_embs[0].num_embeddings == 1024
assert m.input_proj.out_features == 512
print('PASS: module registration OK')
"
```

```bash
uv run python -c "
import torch
from models.tts.comelsinger.svt_module import SVTModule
m = SVTModule()
x = torch.randint(0, 1024, (2, 50, 12))
out = m._embed(x)
assert out.shape == (2, 50, 512), f'expected (2,50,512), got {out.shape}'
print('PASS: _embed shape OK')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.svt_module import SVTModule
m = SVTModule()
n_embed = sum(p.numel() for p in m.codebook_embs.parameters())
n_proj  = sum(p.numel() for p in m.input_proj.parameters())
print(f'codebook_embs params: {n_embed}')   # 12*1024*64 = 786,432
print(f'input_proj params: {n_proj}')       # 768*512+512 = 393,728
assert n_embed == 12 * 1024 * 64
assert n_proj  == 768 * 512 + 512
print('PASS: param count OK')
"
```

## 5. 懸念事項とレビュー項目

- **メモリ効率**: `torch.cat` で一時テンソル `(B, L, 768)` が生成される。大バッチでは `input_proj` を `_embed` 内でインプレース適用しても削減できないため、バッチサイズで調整する。
- **codebook 順序**: M2-09 の `collate_fn` が `acoustic_tokens` を `(B, 12, T)` で返す場合、ここでは `(B, T, 12)` への転置が必要。インターフェースを統一すること。

### レビュー項目

- [ ] `len(m.codebook_embs) == 12` か
- [ ] `_embed` の出力 shape が `(B, L, 512)` か
- [ ] `input_proj` の `in_features == 768`（=12×64）か
- [ ] `codebook_embs` が `nn.ModuleList` として登録され勾配が計算されるか

## 6. フェーズ振り返り: 一から作り直すとしたら

**Transformer vs Conformer**: 音響トークンは時系列データであり、畳み込みによる局所パターン抽出が有効。Conformer ブロックを使う場合、`input_proj` の後に Depth-wise Conv1d を挟む設計が自然。骨格段階で `conv_kernel_size` オプションを用意しておく。

**encoder-only vs decoder**: `_embed` の設計は双方向にも一方向にも対応可能。骨格段階で `causal` フラグを `__init__` に追加し、`input_proj` の後に `causal_mask` を生成するオプションを持たせる。

**位置エンコーディング選択**: `_embed` の後に位置エンコーディングを加算する設計（M2-03）。`input_proj` の出力次元 512 と PosEmb 次元が一致することを M2-03 実装者が確認すること。

## 7. 後続タスクへの連絡事項

- **M2-03**: `_embed` の出力 `(B, L, 512)` に位置エンコーディング `(L, 512)` をブロードキャスト加算する。次元が一致していることを確認すること。
- **M2-09**: `collate_fn` が `acoustic_tokens` を `(B, 12, T)` 形状で返す場合、`SVTModule.forward` 内で `.permute(0, 2, 1)` して `(B, T, 12)` に変換すること。
- **M2-06**: `codebook_embs.parameters()` に勾配が流れることを統合テストで確認すること（SVT 単独学習時は requires_grad=True）。
