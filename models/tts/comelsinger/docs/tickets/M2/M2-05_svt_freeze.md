# M2-05: SVTModule freeze / unfreeze ユーティリティ実装

> **マイルストーン**: [M2: コアモジュール](../../13_milestones.md#m2-コアモジュール実装)
> **対応RQ**: RQ-02
> **依存チケット**: M2-04
> **ブロックするチケット**: M2-06
> **状態**: TODO

---

## 1. 目的とゴール

`SVTModule` に `freeze()` / `unfreeze()` ユーティリティを実装する。S2A 学習時に SVT を StopGrad（frozen）として使用するため、`freeze()` 呼び出し後は全パラメータの `requires_grad=False` かつ `eval()` モードになることを保証する。

`unfreeze()` は SVT 単独学習フェーズへの切り替えに使用する。

## 2. 実装する内容の詳細

```python
# svt_module.py SVTModule に追加

def freeze(self) -> "SVTModule":
    """全パラメータを凍結し eval モードへ切り替える。
    S2A 学習時の StopGrad として使用する。
    Returns self for method chaining.
    """
    self.requires_grad_(False)
    self.eval()
    return self

def unfreeze(self) -> "SVTModule":
    """全パラメータを学習可能に戻し train モードへ切り替える。
    SVT 単独学習フェーズで使用する。
    Returns self for method chaining.
    """
    self.requires_grad_(True)
    self.train()
    return self

def is_frozen(self) -> bool:
    """全パラメータが凍結されているか確認する。"""
    return not any(p.requires_grad for p in self.parameters())
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `freeze`, `unfreeze`, `is_frozen` の実装 |

## 4. 提供範囲とテスト項目

**含むもの**: `freeze`, `unfreeze`, `is_frozen` メソッドの実装

**含まないもの**: 損失計算（→ M2-06）、S2A 側での呼び出しコード（→ M2-11〜M2-15）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.svt_module import SVTModule
m = SVTModule()
m.freeze()
assert m.is_frozen(), 'should be frozen after freeze()'
assert not m.training, 'should be in eval mode after freeze()'
print('PASS: freeze() OK')
"
```

```bash
uv run python -c "
import torch
from models.tts.comelsinger.svt_module import SVTModule
m = SVTModule()
m.freeze()
m.unfreeze()
assert not m.is_frozen(), 'should not be frozen after unfreeze()'
assert m.training, 'should be in train mode after unfreeze()'
print('PASS: unfreeze() OK')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.svt_module import SVTModule
m = SVTModule()
m.freeze()
x = torch.randint(0, 1024, (1, 10, 12))
out = m(x)
# frozen 状態で forward が動作し、かつ勾配が計算されないことを確認
assert out['logits'].requires_grad == False, 'logits should not require grad when frozen'
print('PASS: frozen forward OK, no grad computed')
"
```

## 5. 懸念事項とレビュー項目

- **Batch Normalization の挙動**: `freeze()` で `eval()` を呼ぶため BatchNorm が推論統計を使用する。SVTModule には BatchNorm がないため問題ないが、将来追加する場合は注意する。
- **メソッドチェーン**: `freeze()` / `unfreeze()` は `self` を返すため `model.freeze().some_method()` が可能。これを S2A 初期化コードで活用すること。

### レビュー項目

- [ ] `freeze()` 後に `any(p.requires_grad for p in m.parameters())` が `False` か
- [ ] `freeze()` 後に `m.training` が `False` か
- [ ] `unfreeze()` 後にすべてのパラメータが `requires_grad=True` か
- [ ] `is_frozen()` が状態を正しく返すか

## 6. フェーズ振り返り: 一から作り直すとしたら

**Transformer vs Conformer**: `freeze()` の実装はアーキテクチャに依存しない（`self.requires_grad_()` はすべての submodule に再帰的に適用される）。Conformer に変更しても本チケットの変更は不要。

**encoder-only vs decoder**: `freeze()` / `unfreeze()` はアーキテクチャに依存しないため変更不要。

**位置エンコーディング選択**: `SinusoidalPosEmb` が `nn.Module` として登録されている場合、`freeze()` でパラメータ（あるとすれば）も凍結される。`SinusoidalPosEmb` が parameter-free であれば影響なし。

## 7. 後続タスクへの連絡事項

- **M2-06**: `freeze()` 後に `loss.backward()` を実行した際、SVT のパラメータに勾配が流れないことを統合テストで確認すること。
- **M2-11（CoMelSinger_S2A 骨格）**: `__init__` 内で `svt_module.freeze()` を呼び出す設計を想定している。初期化直後に frozen 状態になることを前提とする。
- **M3（S2A 学習ループ）**: エポック開始時に `svt_module.is_frozen()` でアサーションを入れることを推奨する。誤って `unfreeze()` されていた場合の早期検出のため。
