# M3-11: Algorithm 1 プロンプト生成

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07b
> **依存チケット**: M3-10
> **ブロックするチケット**: M3-12
> **状態**: TODO

---

## 1. 目的とゴール

`prompt_gen()` 関数を実装する。同一話者IDを持つバッチ内の別サンプルを音色プロンプトとして選択する。対照学習でprosody leakageを防ぐため、プロンプトはターゲットとは異なるフレーズから取得する必要がある。同一話者の別サンプルが存在しない場合はランダムな別サンプルで代替する。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/alg1_utils.py に追加
from typing import Dict
import torch

def prompt_gen(batch: Dict[str, torch.Tensor], max_len: int = 150) -> Dict[str, torch.Tensor]:
    """同一話者の異なるサンプルをプロンプトとして選択する。

    Args:
        batch: バッチdict。'speaker_id' キーを含むこと
        max_len: プロンプトの最大フレーム長

    Returns:
        prompt_batch: 各サンプルに対応するプロンプトを含むdict
    """
    speaker_ids = batch["speaker_id"]  # (B,)
    B = speaker_ids.size(0)
    prompt_indices = torch.zeros(B, dtype=torch.long)

    for i in range(B):
        same_spk = (speaker_ids == speaker_ids[i]).nonzero(as_tuple=True)[0]
        candidates = same_spk[same_spk != i]
        if len(candidates) > 0:
            prompt_indices[i] = candidates[torch.randint(len(candidates), (1,))]
        else:
            other = list(range(B)); other.remove(i)
            prompt_indices[i] = other[torch.randint(len(other), (1,)).item()]

    prompt_batch = {k: v[prompt_indices, :max_len] if v.dim() > 1 else v[prompt_indices]
                    for k, v in batch.items()}
    return prompt_batch
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | prompt_gen()実装・同一話者選択ロジック・max_len truncation |

## 4. 提供範囲とテスト項目

**含むもの**: `prompt_gen()` 関数（`alg1_utils.py` に追加）、同一話者選択、max_len truncation

**含まないもの**: S2Aへの統合（→ M3-12）、音声エンコーダへの入力変換

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.alg1_utils import prompt_gen

# 同一話者から異なるインデックスが選ばれること
batch = {
    'acoustic_tokens': torch.zeros(8, 200, 8, dtype=torch.long),
    'speaker_id': torch.tensor([0,0,0,0,1,1,1,1]),
}
prompt = prompt_gen(batch)
print('PASS: prompt_gen runs without error')
assert prompt['acoustic_tokens'].shape[1] <= 150
print('PASS: max_len truncation OK')
"
```

```bash
uv run python -c "
import torch
from models.tts.comelsinger.alg1_utils import prompt_gen

# 話者が1人のバッチでも動作すること（ランダム代替）
batch = {
    'acoustic_tokens': torch.zeros(4, 100, 8, dtype=torch.long),
    'speaker_id': torch.zeros(4, dtype=torch.long),
}
prompt = prompt_gen(batch)
print('PASS: single speaker fallback OK')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.alg1_utils import prompt_gen

batch = {
    'acoustic_tokens': torch.zeros(32, 200, 8, dtype=torch.long),
    'speaker_id': torch.randint(0, 5, (32,)),
}
prompt = prompt_gen(batch)
assert prompt['acoustic_tokens'].shape == (32, 150, 8)
print('PASS: prompt_gen with 32-batch and 5 speakers OK')
"
```

## 5. 懸念事項とレビュー項目

- **計算コスト**: B回のfor loopは小さいバッチでは問題ないが、大きいバッチでは vectorize を検討する。
- **プロンプト長の統一**: max_len=150フレーム（約6秒@25fps）は論文仕様を確認すること。固定長パディングとどちらが良いか検討する。

### レビュー項目

- [ ] 同一話者の別インデックスが選ばれていること（i != prompt_indices[i] が保証されること）
- [ ] prompt の acoustic_tokens の第2次元が max_len 以下であること
- [ ] 話者が1人のバッチで例外なく動作すること

## 6. フェーズ振り返り: 一から作り直すとしたら

> 共通の設計判断（PyTorch Lightning vs Accelerate、実験管理、スケジューラ選択）は [M3_design_decisions.md](M3_design_decisions.md) を参照のこと。以下はこのチケット固有の設計判断を記載する。


- **PyTorch Lightning vs 素のAccelerate**: この関数はフレームワーク非依存だが、DataLoaderにCustomSamplerを組み込んでバッチ構成時点でプロンプトペアを準備する設計が効率的。
- **実験管理(W&B/MLflow)**: プロンプト選択の話者分布をW&Bに記録し、バッチ内の話者バランスを可視化できる仕組みを入れる。
- **Algorithm 1の関数分解粒度**: `split_batch → pitch_perturbation → prompt_gen` を一つの `prepare_s2a_batch()` として統合することで、学習ループのコードが読みやすくなる。

## 7. 後続タスクへの連絡事項

- **M3-12**: `prompt_gen()` の出力（`prompt_batch`）は S2A の `cond_emb` 計算に使用する。`acoustic_tokens` キーをCoMelSinger_S2Aの参照音声エンコーダに渡す実装を確認すること。
- **M3-13**: SCL計算では `s_a_s`（8件）とその対応プロンプト（8件）を使用する。prompt_genの出力を`s_a_s`に対して実行することに注意。
- **M3-14**: FCL計算では `s_a_f`（24件）とそのプロンプトを使用する。プロンプト生成はsplit後に各セットに対して独立して実行する。
