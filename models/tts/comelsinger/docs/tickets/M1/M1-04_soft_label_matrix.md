# M1-04: compute_soft_label_matrix 実装

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-01
> **依存チケット**: M1-01
> **ブロックするチケット**: M1-06
> **状態**: TODO

---

## 1. 目的とゴール

`PitchTokenizer.compute_soft_label_matrix()` を実装する。フレームアラインドなピッチトークン列 `m_p (L,)` を受け取り、FCL（フレームレベル対照学習）で使用するソフトラベル行列 `Y (L,L)` を生成する。

`Y[i,j]=1.0` は「フレーム i と j が同一ピッチであり、かつ両方有声」であることを示す正例ペア。無声フレーム同士（token=0）は `Y[i,j]=0.0` とし、正例としない。

## 2. 実装する内容の詳細

### 2.1 アルゴリズム詳細

```
入力: m_p (L,) — ピッチトークン列, dtype=long
  token 0 = 無声フレーム
  token 1-128 = 有声フレーム（MIDI番号）

処理:
  voiced_mask = (m_p != 0)           # (L,) bool
  voiced_2d = voiced_mask[:, None] & voiced_mask[None, :]  # (L,L) 両方有声
  same_pitch = (m_p[:, None] == m_p[None, :])              # (L,L) 同一ピッチ
  Y = (voiced_2d & same_pitch).float()                     # (L,L)

出力: Y (L,L) float, 値域 {0.0, 1.0}
```

### 2.2 実装コード

```python
def compute_soft_label_matrix(
    self, m_p: torch.LongTensor
) -> torch.FloatTensor:
    """
    (L,) → (L,L) ソフトラベル行列

    Args:
        m_p: ピッチトークン列 (L,), dtype=long
             0=無声, 1-128=MIDI番号

    Returns:
        Y: ソフトラベル行列 (L,L), dtype=float
           Y[i,j]=1.0: フレーム i,j が同一ピッチ & 両方有声
           Y[i,j]=0.0: それ以外（無声含む）
    """
    voiced_mask = (m_p != 0)                                    # (L,)
    voiced_2d = voiced_mask.unsqueeze(1) & voiced_mask.unsqueeze(0)  # (L,L)
    same_pitch = (m_p.unsqueeze(1) == m_p.unsqueeze(0))         # (L,L)
    Y = (voiced_2d & same_pitch).float()                         # (L,L)
    return Y
```

### 2.3 行列の特性

| 特性 | 値 |
|---|---|
| 対角成分 `Y[i,i]` | 有声フレームなら 1.0、無声なら 0.0 |
| 対称性 | `Y == Y.T` を常に満たす |
| 無声同士 | `Y[i,j]=0.0`（token 0 同士でも正例としない） |
| 有声・無声混在 | `Y[i,j]=0.0` |
| 全無声入力 | `Y` が全 0 の行列 |

### 2.4 FCL での使用方法

```python
# compute_fcl_loss() での使用例
# m_p_a: バッチAのピッチトークン (B, L)
# 各バッチサンプルに対してソフトラベル行列を計算

Y = torch.stack([
    tokenizer.compute_soft_label_matrix(m_p_a[b])
    for b in range(B)
])  # (B, L, L)
```

**重要: Python は全て `uv run python` / `uv run pytest` で実行。pip 禁止。**

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `compute_soft_label_matrix` メソッドの実装 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: `compute_soft_label_matrix` メソッドの完全実装

**含まないもの**: テストファイル作成（→ M1-06）、FCL 損失関数（→ M1-08）

### 4.2 ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer
tok = PitchTokenizer()

# 基本ケース: 有声・同一ピッチ
m_p = torch.tensor([69, 69, 72, 0], dtype=torch.long)
Y = tok.compute_soft_label_matrix(m_p)

# 対称性
assert (Y == Y.T).all(), 'not symmetric'
print('PASS: symmetry')

# 有声同士同一ピッチ = 1.0
assert Y[0, 1].item() == 1.0, f'expected 1.0, got {Y[0,1].item()}'
print('PASS: voiced same pitch = 1.0')

# 異なるピッチ = 0.0
assert Y[0, 2].item() == 0.0, f'expected 0.0, got {Y[0,2].item()}'
print('PASS: different pitch = 0.0')

# 無声同士 = 0.0 (正例としない)
m_p2 = torch.tensor([0, 0], dtype=torch.long)
Y2 = tok.compute_soft_label_matrix(m_p2)
assert Y2[0, 1].item() == 0.0, f'expected 0.0, got {Y2[0,1].item()}'
print('PASS: unvoiced pair = 0.0')

# 有声と無声の混在 = 0.0
assert Y[0, 3].item() == 0.0, f'expected 0.0, got {Y[0,3].item()}'
print('PASS: voiced-unvoiced = 0.0')
"
```

### 4.3 E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer
tok = PitchTokenizer()

# 全無声 → 全ゼロ行列
m_p = torch.zeros(10, dtype=torch.long)
Y = tok.compute_soft_label_matrix(m_p)
assert Y.sum().item() == 0.0, 'all-unvoiced should be all-zero'
print('PASS: all-unvoiced -> all-zero')

# 全有声同一ピッチ → 全1行列
m_p = torch.full((5,), 69, dtype=torch.long)
Y = tok.compute_soft_label_matrix(m_p)
assert Y.sum().item() == 25.0, f'expected 25.0, got {Y.sum().item()}'
print('PASS: all-voiced same-pitch -> all-one')

# 形状確認
m_p = torch.randint(0, 129, (100,))
Y = tok.compute_soft_label_matrix(m_p)
assert Y.shape == (100, 100), f'expected (100,100), got {Y.shape}'
print('PASS: shape (L,L)')
"
```

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- **無声同士の扱い**: 無声フレーム（token=0）が全て同じ値であるため、単純に `same_pitch` だけで判定すると無声同士が全て正例になってしまう。`voiced_2d` との AND が必須。
- **バッチ次元**: このメソッドは `(L,)` を受け取るシングルサンプル処理。`(B, L)` のバッチ処理は呼び出し側で `for b in range(B)` または `vmap` で対処する。
- **メモリ使用量**: `(L,L)` の行列は L=1000 の場合 4MB（float32）。長い系列では注意が必要。

### 5.2 レビュー項目

- [ ] 対称性 `Y == Y.T` が成立するか
- [ ] 無声同士 `(token=0, token=0)` が `Y[i,j]=0.0` になるか
- [ ] 有声・同一ピッチが `Y[i,j]=1.0` になるか
- [ ] 有声・異なるピッチが `Y[i,j]=0.0` になるか
- [ ] 形状が `(L,L)` になるか

## 6. フェーズ振り返り: 一から作り直すとしたら

**ソフトラベルの連続値化**

現在の実装は `Y ∈ {0.0, 1.0}` のハードラベルだが、論文では「soft label」と呼んでいる。これはピッチのパディング/繰り返し（音符持続中の連続フレームへの対応）に言及しており、「近いピッチほど正例に近い値」とする連続化が本来の意図の可能性がある。

一から設計するなら、`Y[i,j] = exp(-|token_i - token_j|^2 / (2*sigma^2))` のようなガウスカーネルによるソフトラベルも実験対象として `sigma` パラメータで切り替えられる設計にする。

**対称ブロック行列での高速化**

同一ピッチの区間は連続フレームが多いため、`(i, j)` ペアの多くが同一の Y 値を持つ。スパース表現または Run-Length Encoding により `(L,L)` の密行列を生成せずに損失計算できる可能性があるが、PyTorch のブロードキャスト計算で十分高速であるため、現状では密行列で実装する。

**型安全性**

入力を `torch.LongTensor` に限定しているが、実際には `torch.IntTensor` が来るケースも考えられる。`m_p = m_p.long()` の変換を冒頭で追加することを推奨する。

## 7. 後続タスクへの連絡事項

- **M1-06 (test_pitch_tokenizer)**: 対称性テスト・無声ペアの非正例確認・形状確認の3項目は必ずテストケースに含めること。
- **M1-08 (compute_fcl_loss)**: Y 行列は `{0.0, 1.0}` の二値（ハードラベル）として提供する。論文の "soft label" という用語はピッチのパディング/繰り返し対応を意味しており、連続値（ガウスカーネル等）ではなくピッチ一致/不一致の二値を指す。FCL 側で行正規化等の追加処理が必要かを実装時に検討すること。また、`Y` がバッチ次元 `(B,L,L)` で渡されることを想定すること。
- **M1-11 (test_losses)**: FCL テストでは `Y` 全ゼロ（全無声）の場合に `loss==0` になることを確認すること（M1-08 との統合動作確認）。
