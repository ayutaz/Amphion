# M2-13: S2A 推論用 get_cond ヘルパー実装

> **マイルストーン**: [M2: コアモジュール](../../13_milestones.md#m2-コアモジュール実装)
> **対応RQ**: RQ-03
> **依存チケット**: M2-12
> **ブロックするチケット**: M2-14
> **状態**: TODO

---

## 1. 目的とゴール

`CoMelSinger_S2A` に `get_cond(cond_code, pitch_tokens)` → `(B, T, 1024)` ヘルパーを実装する。推論パイプライン（`maskgct_inference.py` の派生）から呼び出され、cond 埋め込みとピッチ埋め込みを合算した条件テンソルを返す。

推論時に `pitch_tokens=None` を渡した場合は cond 埋め込みのみを返す（TTS モード互換）。

## 2. 実装する内容の詳細

```python
# comelsinger_s2a.py CoMelSinger_S2A に追加
import torch

def get_cond(
    self,
    cond_code: torch.LongTensor,
    pitch_tokens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """条件テンソルを生成する推論用ヘルパー。

    Args:
        cond_code:     (B, T) 参照音声の cond コード（セマンティックトークン等）
        pitch_tokens:  (B, T) ピッチトークン。None の場合はピッチ条件なし。
    Returns:
        cond: (B, T, 1024) 条件埋め込みテンソル
    """
    # 親クラスの cond_emb を使用（実装確認が必要）
    cond = self.cond_emb(cond_code)           # (B, T, 1024)
    if pitch_tokens is not None:
        cond = cond + self.pitch_emb(pitch_tokens)  # (B, T, 1024)
    return cond
```

注意: 親クラスの `cond_emb` の名前・API は `maskgct_s2a.py` を確認してから実装すること。

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `get_cond` メソッドの実装 |

## 4. 提供範囲とテスト項目

**含むもの**: `get_cond` メソッドの実装

**含まないもの**: 推論パイプライン本体（→ M4）、マスク損失統合（→ M2-14）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
model = CoMelSinger_S2A()
B, T = 1, 100
cond_code = torch.randint(0, 1024, (B, T))
pitch_tokens = torch.randint(0, 129, (B, T))
cond = model.get_cond(cond_code, pitch_tokens)
assert cond.shape == (B, T, 1024), f'expected (1,100,1024), got {cond.shape}'
print('PASS: get_cond with pitch OK')
"
```

```bash
uv run python -c "
import torch
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
model = CoMelSinger_S2A()
B, T = 1, 100
cond_code = torch.randint(0, 1024, (B, T))
# pitch_tokens=None でも動作するか
cond = model.get_cond(cond_code, pitch_tokens=None)
assert cond.shape == (B, T, 1024)
print('PASS: get_cond without pitch OK')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
model = CoMelSinger_S2A()
model.eval()
B, T = 1, 100
cond_code = torch.randint(0, 1024, (B, T))
pitch_tokens = torch.randint(0, 129, (B, T))
# with/without pitch の差分確認
cond_with    = model.get_cond(cond_code, pitch_tokens)
cond_without = model.get_cond(cond_code, None)
diff = (cond_with - cond_without).abs().mean().item()
assert diff > 0, 'pitch should change cond'
print(f'PASS: pitch changes cond (mean diff={diff:.4f})')
"
```

## 5. 懸念事項とレビュー項目

- **`self.cond_emb` の存在確認**: 親クラス `MaskGCT_S2A` に `cond_emb` が定義されているか `maskgct_s2a.py` で確認すること。名前が異なる場合は適切な属性名を使用する。
- **推論モード（`eval()`）**: `get_cond` は推論専用のため、`torch.no_grad()` コンテキスト内で呼び出されることを前提とする。`forward` との違いを明確にコメントすること。

### レビュー項目

- [ ] `get_cond(code, pitch).shape == (B, T, 1024)` か
- [ ] `get_cond(code, None).shape == (B, T, 1024)` か
- [ ] `pitch_tokens` の有無で `cond` の値が変わるか
- [ ] `self.cond_emb` が `nn.Module` として登録されているか

## 6. フェーズ振り返り: 一から作り直すとしたら

**継承 vs コンポジション**: `get_cond` は `self.cond_emb` を直接参照するため、親クラスの内部実装に依存している。コンポジション設計なら `self.base_model.cond_emb(...)` として依存を明示できる。

**LoRA vs QLoRA vs フルFT**: `cond_emb` が LoRA の対象外（M2-15 では `q_proj`, `v_proj` のみ）であるため、`get_cond` の `cond_emb` 呼び出しは学習/推論で同一の挙動になる。LoRA 適用後に `get_cond` の出力が変わらないことを M2-15 で確認すること。

**ハイパーパラメータ管理**: `embedding_dim=1024` の仮定を `get_cond` 内に持たせないことが重要。`self.cond_emb.embedding_dim` を動的に取得する設計にする。

## 7. 後続タスクへの連絡事項

- **M2-14**: `get_cond` が返す `(B, T, 1024)` テンソルを `compute_loss` に渡す形で統合テストを実装すること。
- **M4（推論パイプライン）**: `maskgct_inference.py` の派生クラスで `model.get_cond(cond_code, pitch_tokens)` を呼び出してから反復デコーディングを行う設計を想定している。
- **M3（学習ループ）**: `get_cond` は推論専用。学習時は `forward` 内で直接 `pitch_emb` を適用するため、`get_cond` は使わないこと。
