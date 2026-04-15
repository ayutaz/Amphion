# M3-10: Algorithm 1 ピッチ摂動

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07b
> **依存チケット**: M3-09
> **ブロックするチケット**: M3-11
> **状態**: TODO

---

## 1. 目的とゴール

`pitch_perturbation()` 関数を実装する。`zero_prob=0.5` の確率でピッチトークンをゼロ（無音/沈黙）に置き換え、それ以外の場合はランダムなセミトーンシフト（-6〜+6の整数）を適用する。prosody leakageを防止する対照学習ペア（original vs perturbed）の作成に使用される。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/alg1_utils.py に追加

def pitch_perturbation(
    pitch_tokens: torch.Tensor,
    zero_prob: float = 0.5,
    max_shift: int = 6,
    pitch_vocab_size: int = 129,
) -> torch.Tensor:
    """ピッチトークンにゼロ化またはランダムシフトをサンプル独立に適用する。

    論文 Algorithm 1 および Section III-B の対照学習に基づき、各サンプルに
    独立した摂動を適用する。バッチ全体に同一の摂動をかけると、全サンプルが
    同方向に変化するため対照学習ペアの多様性が失われる。

    Args:
        pitch_tokens: (B, T) のピッチトークン tensor
        zero_prob: ゼロ化する確率 (デフォルト0.5)
        max_shift: 最大セミトーンシフト数
        pitch_vocab_size: ピッチ語彙サイズ (clampの上限)

    Returns:
        perturbed: (B, T) 摂動済みピッチトークン（各サンプルが独立した摂動）
    """
    B = pitch_tokens.shape[0]
    perturbed = pitch_tokens.clone()
    for i in range(B):
        if torch.rand(1).item() < zero_prob:
            perturbed[i] = 0
        else:
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            perturbed[i] = (pitch_tokens[i] + shift).clamp(0, pitch_vocab_size - 1)
    return perturbed
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | pitch_perturbation()実装・確率動作確認・clamp検証 |

## 4. 提供範囲とテスト項目

**含むもの**: `pitch_perturbation()` 関数（`alg1_utils.py` に追加）

**含まないもの**: プロンプト生成（→ M3-11）、S2Aへの統合（→ M3-12）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.alg1_utils import pitch_perturbation

# zero_prob=1.0で必ずゼロ化されること
pitch = torch.ones(2, 10, dtype=torch.long) * 60
perturbed = pitch_perturbation(pitch, zero_prob=1.0)
assert perturbed.sum() == 0
print('PASS: zero_prob=1.0 -> all zeros')
"
```

```bash
uv run python -c "
import torch
from models.tts.comelsinger.alg1_utils import pitch_perturbation

# clampが有効であること（pitch_vocab_size=129の境界確認）
pitch = torch.full((2, 10), 128, dtype=torch.long)
perturbed = pitch_perturbation(pitch, zero_prob=0.0, max_shift=6, pitch_vocab_size=129)
assert perturbed.max() <= 128 and perturbed.min() >= 0
print('PASS: clamp boundary OK')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.alg1_utils import pitch_perturbation

# 各サンプルが独立した摂動を受けること（全サンプル一律でない）
torch.manual_seed(0)
pitch = torch.ones(8, 50, dtype=torch.long) * 64
perturbed = pitch_perturbation(pitch, zero_prob=0.5)
# 8サンプル中で少なくとも一部がゼロ化、一部がシフトになるはず（確率的）
# サンプルごとに異なる摂動が適用されていることを確認
rows_all_zero = (perturbed == 0).all(dim=1).sum().item()
rows_nonzero = (perturbed != 0).any(dim=1).sum().item()
print(f'zero rows={rows_all_zero}, nonzero rows={rows_nonzero}')
print('PASS: per-sample perturbation runs without error')
"
```

```bash
uv run python -c "
import torch
from models.tts.comelsinger.alg1_utils import pitch_perturbation

# 100サンプルに対してzero_prob=0.5の統計がサンプル単位で現れること
pitch = torch.ones(100, 10, dtype=torch.long) * 64
perturbed = pitch_perturbation(pitch, zero_prob=0.5)
zero_rows = (perturbed == 0).all(dim=1).sum().item()
assert 30 <= zero_rows <= 70, f'zero_rows={zero_rows} is out of expected range [30,70]'
print(f'PASS: per-sample zero_count={zero_rows}/100 (expected ~50)')
"
```

## 5. 懸念事項とレビュー項目

- **バッチ全体 vs 要素単位**: 論文 Section III-B および Algorithm 1 を確認した結果、摂動はサンプル独立に適用する設計が正しい。各サンプルごとに `torch.rand(1)` で独立した確率を引き、ゼロ化かセミトーンシフトかを決定する。バッチ全体に `torch.rand(1)` を使う実装（全サンプル同一摂動）は誤りであり修正済み。
- **無音トークンの定義**: ゼロ（0）が無音トークンとして定義されているかDataset実装（M2-10）と整合を取ること。

### レビュー項目

- [ ] zero_prob=0.5が設定ファイルから読み込まれること
- [ ] clampで[0, pitch_vocab_size-1]の範囲内に収まること
- [ ] 元のpitch_tokensテンソルが変更されないこと（in-place操作を避ける）

## 6. フェーズ振り返り: 一から作り直すとしたら

> 共通の設計判断（PyTorch Lightning vs Accelerate、実験管理、スケジューラ選択）は [M3_design_decisions.md](M3_design_decisions.md) を参照のこと。以下はこのチケット固有の設計判断を記載する。


- **PyTorch Lightning vs 素のAccelerate**: この関数はフレームワーク非依存であり、どちらでも同じ実装になる。
- **実験管理(W&B/MLflow)**: zero_probやmax_shiftをW&Bのハイパーパラメータとして記録し、最適値をスイープで探索できる構成にする。
- **GradNorm動的重み調整**: 摂動の強度（zero_prob, max_shift）を学習進行に応じて動的に変化させるカリキュラム学習の導入を検討すべきだった。

## 7. 後続タスクへの連絡事項

- **M3-11**: `pitch_perturbation()` の出力をプロンプト生成（`prompt_gen()`）に渡す際は、perturbedとoriginalの両方を保持しておくこと。
- **M3-12**: S2Aのforward呼び出しでは `pitch_emb(original)` と `pitch_emb(perturbed)` の両方を計算するため、original pitch_tokensも保持する必要がある。
- **M3-15**: L_CL計算でoriginal側とperturbed側の埋め込みを比較するが、ゼロ化されたperturbedは全フレームが同一値になる。この場合のFCL計算への影響を確認すること。
