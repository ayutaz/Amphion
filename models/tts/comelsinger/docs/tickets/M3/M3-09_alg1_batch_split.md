# M3-09: Algorithm 1 バッチ分割

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07b
> **依存チケット**: M3-08
> **ブロックするチケット**: M3-10
> **状態**: TODO

---

## 1. 目的とゴール

バッチB（K=32サンプル）をS_A_s（acoustic先頭8件: インデックス[:8]）とS_A_f（残り24件: インデックス[8:]）に分割する `split_batch()` 関数を実装する。S_A_sはSCL（シーケンスレベル対照学習）用、S_A_fはFCL（フレームレベル対照学習）用として使用される。K_s=8は設定ファイルから読み込む。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/alg1_utils.py
from typing import Dict
import torch

def split_batch(batch: Dict[str, torch.Tensor], K_s: int = 8):
    """バッチをSCL用(S_A_s)とFCL用(S_A_f)に分割する。

    Args:
        batch: DataLoaderから得られたバッチdict (B=K_s + N_f サンプル)
        K_s: SCL用サンプル数 (デフォルト8)

    Returns:
        s_a_s: dict (上位K_s件) - SCL用
        s_a_f: dict (残り) - FCL用
    """
    s_a_s = {k: v[:K_s] for k, v in batch.items()}
    s_a_f = {k: v[K_s:] for k, v in batch.items()}
    return s_a_s, s_a_f
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | split_batch()実装・スライス動作確認 |

## 4. 提供範囲とテスト項目

**含むもの**: `models/tts/comelsinger/alg1_utils.py` 新規作成、`split_batch()` 関数

**含まないもの**: ピッチ摂動（→ M3-10）、プロンプト生成（→ M3-11）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.alg1_utils import split_batch

batch = {
    'acoustic_tokens': torch.zeros(32, 100, 8, dtype=torch.long),
    'pitch_tokens': torch.zeros(32, 100, dtype=torch.long),
    'speaker_id': torch.zeros(32, dtype=torch.long),
}
s_a_s, s_a_f = split_batch(batch, K_s=8)
assert s_a_s['acoustic_tokens'].shape == (8, 100, 8)
assert s_a_f['acoustic_tokens'].shape == (24, 100, 8)
print('PASS: split_batch shapes OK')
"
```

```bash
uv run python -c "
import torch
from models.tts.comelsinger.alg1_utils import split_batch

batch = {'x': torch.arange(32)}
s_a_s, s_a_f = split_batch(batch, K_s=8)
assert list(s_a_s['x'].numpy()) == list(range(8))
assert list(s_a_f['x'].numpy()) == list(range(8, 32))
print('PASS: split_batch indices OK')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.alg1_utils import split_batch

# K_s=8でバッチを分割し、全キーが正しく分割されること
batch = {k: torch.randn(32, 10) for k in ['acoustic_tokens', 'pitch_tokens', 'semantic_tokens']}
s, f = split_batch(batch, K_s=8)
assert all(s[k].shape[0] == 8 for k in s)
assert all(f[k].shape[0] == 24 for k in f)
print('PASS: all keys split correctly')
"
```

## 5. 懸念事項とレビュー項目

- **バッチサイズの前提**: batch_size=32, K_s=8が前提。バッチが32未満の場合（データセット末尾など）は要対処。drop_last=Trueを推奨する。
- **dictキーの網羅**: `batch` の全キーを `s_a_s` と `s_a_f` に分割すること。特定キーのみ選択する実装は避ける。

### レビュー項目

- [ ] s_a_s のサイズが K_s=8 であること
- [ ] s_a_f のサイズが batch_size - K_s であること
- [ ] 全バッチキーが両方のdictに含まれること

## 6. フェーズ振り返り: 一から作り直すとしたら

> 共通の設計判断（PyTorch Lightning vs Accelerate、実験管理、スケジューラ選択）は [M3_design_decisions.md](M3_design_decisions.md) を参照のこと。以下はこのチケット固有の設計判断を記載する。


- **PyTorch Lightning vs 素のAccelerate**: Lightningでは `training_step` 内でこの分割を行う。データフローがより明示的になる。
- **実験管理(W&B/MLflow)**: K_sをハイパーパラメータとしてW&Bに記録することで、K_sの最適値探索が容易になる。
- **Algorithm 1の関数分解粒度**: split_batch, pitch_perturbation, prompt_gen を1つの `prepare_batch()` 関数に統合する設計も検討に値する。個別関数の方がテストは容易。

## 7. 後続タスクへの連絡事項

- **M3-10**: `s_a_s` と `s_a_f` それぞれに対してピッチ摂動を適用する。`s_a_s` のみ摂動版を作成する設計とする（SCL用にoriginalとperturbedのペアが必要なため）。
- **M3-13**: SCL計算で使用するのは `s_a_s`（8件）のみ。`K_s=8` の値をSCL計算の入力次元として引き継ぐ。
- **M3-14**: FCL計算で使用するのは `s_a_f`（24件）のみ。`s_a_f` のシーケンス長Tが可変のため、パディング対応を確認すること。
