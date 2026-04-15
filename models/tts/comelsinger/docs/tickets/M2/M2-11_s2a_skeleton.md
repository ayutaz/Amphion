# M2-11: CoMelSinger_S2A クラス骨格

> **マイルストーン**: [M2: コアモジュール](../../13_milestones.md#m2-コアモジュール実装)
> **対応RQ**: RQ-03
> **依存チケット**: M1
> **ブロックするチケット**: M2-12
> **状態**: TODO

---

## 1. 目的とゴール

`comelsinger_s2a.py` に `CoMelSinger_S2A(MaskGCT_S2A)` の骨格を作成する。`pitch_vocab_size=129, temperature=0.07`、損失重み `λ_cl=0.5, λ_scl=1.0, λ_fcl=0.1, λ_svt=0.5, λ_mask=0.3` を `__init__` で受け取る。

`isinstance(model, MaskGCT_S2A)` が `True` になることで、既存の MaskGCT 推論パイプラインとの互換性を保ちながら SVS 向け拡張が可能になる。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/comelsinger_s2a.py
from __future__ import annotations
from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A

class CoMelSinger_S2A(MaskGCT_S2A):
    """MaskGCT_S2A を継承した歌唱音声合成 S2A モデル。
    ピッチ埋め込み・対照学習損失・SVT 監督を追加。
    論文 Section III-A, III-B, III-C 対応。
    """

    def __init__(
        self,
        # MaskGCT_S2A に渡すパラメータ（親クラスのシグネチャに合わせて補完）
        *args,
        pitch_vocab_size: int = 129,
        temperature: float = 0.07,
        lambda_cl:   float = 0.5,
        lambda_scl:  float = 1.0,
        lambda_fcl:  float = 0.1,
        lambda_svt:  float = 0.5,
        lambda_mask: float = 0.3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pitch_vocab_size = pitch_vocab_size
        self.temperature = temperature
        self.lambda_cl   = lambda_cl
        self.lambda_scl  = lambda_scl
        self.lambda_fcl  = lambda_fcl
        self.lambda_svt  = lambda_svt
        self.lambda_mask = lambda_mask
        # サブモジュールは後続チケットで追加:
        # self.pitch_emb → M2-12
        # self.svt_module → M2-12 or M2-14
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `comelsinger_s2a.py` 骨格の作成、`__init__` 実装 |

## 4. 提供範囲とテスト項目

**含むもの**: `comelsinger_s2a.py` の新規作成、`CoMelSinger_S2A.__init__` の実装

**含まないもの**: `pitch_emb`（→ M2-12）、`get_cond`（→ M2-13）、損失計算（→ M2-14）

### ユニットテスト

```bash
uv run python -c "
from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
# 親クラスのデフォルト引数でインスタンス化
model = CoMelSinger_S2A()
assert isinstance(model, MaskGCT_S2A), 'must be instance of MaskGCT_S2A'
print('PASS: isinstance check OK')
"
```

```bash
uv run python -c "
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
model = CoMelSinger_S2A()
assert model.pitch_vocab_size == 129
assert model.temperature      == 0.07
assert model.lambda_mask      == 0.3
print('PASS: hyperparameters OK')
"
```

### E2Eテスト

```bash
uv run python -c "
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A
model = CoMelSinger_S2A()
assert issubclass(CoMelSinger_S2A, MaskGCT_S2A)
# 親クラスの forward が呼び出せること（M2-12 前はスタブ可）
print('PASS: class hierarchy OK')
"
```

## 5. 懸念事項とレビュー項目

- **`MaskGCT_S2A.__init__` のシグネチャ確認**: `*args, **kwargs` で渡す前に親クラスのシグネチャを `maskgct_s2a.py` で確認すること。必須引数がある場合はデフォルト値を設定する。
- **ハイパーパラメータの出所**: 論文 Table I に記載の重みと、`CLAUDE.md` の値（`λ_SCL=0.5, λ_FCL=1.0, λ_SVT=0.1, λ_mask=0.3`）の差異を確認すること。

### レビュー項目

- [ ] `isinstance(model, MaskGCT_S2A)` が `True` か
- [ ] 全ハイパーパラメータがインスタンス属性として保持されているか
- [ ] `*args, **kwargs` が親クラスに正しく転送されているか
- [ ] 型ヒントが付与されているか

## 6. フェーズ振り返り: 一から作り直すとしたら

**継承 vs コンポジション**: 継承（現設計）は `isinstance` チェックで互換性を保てるが、親クラスの内部実装変更に脆弱。コンポジション（`self.base = MaskGCT_S2A()`）なら結合度が低いが、`forward` のオーバーライドが煩雑になる。

**LoRA vs QLoRA vs フルFT**: 骨格段階では継承のみ。LoRA の適用は M2-15 で行うが、M2-11 の段階で LoRA 適用を前提としたクラス設計（`target_modules` を引数に持つ）にしておくと拡張が容易。

**ハイパーパラメータ管理**: 損失重み（λ）を `__init__` の引数にする設計はシンプルだが、Hydra/OmegaConf の Config オブジェクトとして外部化するとアブレーション実験が容易になる。

## 7. 後続タスクへの連絡事項

- **M2-12**: `pitch_emb = nn.Embedding(129, 1024)` を `super().__init__()` 呼び出し後に `self` に追加すること。
- **M2-13**: `get_cond` は推論パイプライン用の public メソッドとして定義すること。親クラスの `cond_emb` の存在を確認してから実装する。
- **M2-14**: マスク損失は親クラスの `compute_loss` を呼び出す設計を想定。`super().compute_loss(...)` が動作することを M2-14 で確認すること。
