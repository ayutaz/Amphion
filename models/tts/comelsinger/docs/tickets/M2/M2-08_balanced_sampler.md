# M2-08: BalancedSpeakerSampler 実装

> **マイルストーン**: [M2: コアモジュール](../../13_milestones.md#m2-コアモジュール実装)
> **対応RQ**: RQ-06
> **依存チケット**: M2-07
> **ブロックするチケット**: M2-09
> **状態**: TODO

---

## 1. 目的とゴール

`dataset.py` に `BalancedSpeakerSampler` を実装する。バッチ内の先頭 `k_s=8` サンプルに同一話者ペアが少なくとも 1 ペア含まれることを保証する。SCL（Sequence-level Contrastive Learning）では同一話者ペアが必要なため、このサンプラーが対照学習の品質を左右する。

## 2. 実装する内容の詳細

```python
# dataset.py に追加
from torch.utils.data import Sampler
import random
from collections import defaultdict
from typing import Iterator

class BalancedSpeakerSampler(Sampler):
    """バッチ先頭 k_s サンプルに同一話者ペアを保証するサンプラー。

    アルゴリズム:
    1. 話者ごとのインデックスリストを構築
    2. バッチ生成時: まず anchor として話者をランダム選択
    3. 同一話者から positive を 1 件選択 → 先頭2スロットに配置
    4. 残り (batch_size - 2) スロットをランダム埋め
    5. 先頭 k_s スロット内に同一話者ペアが存在すること保証
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        k_s: int = 8,
        drop_last: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.k_s = k_s
        self.drop_last = drop_last
        # speaker_id → [indices] の逆引き辞書を構築
        self.speaker_to_indices = defaultdict(list)
        for i in range(len(dataset)):
            sid = dataset[i]["speaker_id"]
            self.speaker_to_indices[sid].append(i)
        # 2件以上サンプルがある話者のみペア作成に使用
        self.valid_speakers = [
            s for s, idxs in self.speaker_to_indices.items()
            if len(idxs) >= 2
        ]
        self.all_indices = list(range(len(dataset)))

    def __iter__(self) -> Iterator[list]:
        indices = self.all_indices.copy()
        random.shuffle(indices)
        for start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch = []
            # 先頭にアンカー+ポジティブを配置
            spk = random.choice(self.valid_speakers)
            anchor, positive = random.sample(self.speaker_to_indices[spk], 2)
            batch.extend([anchor, positive])
            # 残りをランダム補充
            rest = [i for i in indices[start:start + self.batch_size] if i not in {anchor, positive}]
            batch.extend(rest[:self.batch_size - 2])
            yield batch[:self.batch_size]

    def __len__(self) -> int:
        n = len(self.all_indices) // self.batch_size
        return n
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `BalancedSpeakerSampler` の実装 |

## 4. 提供範囲とテスト項目

**含むもの**: `BalancedSpeakerSampler` の実装

**含まないもの**: `collate_fn`（→ M2-09）、DataLoader 統合（→ M2-10）

### ユニットテスト

```bash
uv run python -c "
import torch, tempfile
from pathlib import Path
from models.tts.comelsinger.dataset import CoMelSingerDataset, BalancedSpeakerSampler
with tempfile.TemporaryDirectory() as tmpdir:
    (Path(tmpdir) / 'tokens').mkdir()
    for i in range(20):
        torch.save({'acoustic_tokens': torch.zeros(50, 12).long(),
                    'semantic_tokens': torch.zeros(50).long(),
                    'pitch_tokens': torch.zeros(50).long(),
                    'speaker_id': i % 4},
                   Path(tmpdir) / 'tokens' / f's{i}.pt')
    ds = CoMelSingerDataset(tmpdir)
    sampler = BalancedSpeakerSampler(ds, batch_size=8, k_s=8)
    batch = next(iter(sampler))
    spk_ids = [ds[i]['speaker_id'] for i in batch[:8]]
    assert len(set(spk_ids)) < 8, 'should have duplicate speaker in first k_s'
    print('PASS: speaker pair in first k_s OK')
"
```

```bash
uv run python -c "
import torch, tempfile
from pathlib import Path
from models.tts.comelsinger.dataset import CoMelSingerDataset, BalancedSpeakerSampler
with tempfile.TemporaryDirectory() as tmpdir:
    (Path(tmpdir) / 'tokens').mkdir()
    for i in range(16):
        torch.save({'acoustic_tokens': torch.zeros(50, 12).long(),
                    'semantic_tokens': torch.zeros(50).long(),
                    'pitch_tokens': torch.zeros(50).long(),
                    'speaker_id': i % 3},
                   Path(tmpdir) / 'tokens' / f's{i}.pt')
    ds = CoMelSingerDataset(tmpdir)
    sampler = BalancedSpeakerSampler(ds, batch_size=4)
    batches = list(iter(sampler))
    assert all(len(b) == 4 for b in batches)
    print(f'PASS: {len(batches)} batches, all size 4')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch, tempfile
from pathlib import Path
from models.tts.comelsinger.dataset import CoMelSingerDataset, BalancedSpeakerSampler
with tempfile.TemporaryDirectory() as tmpdir:
    (Path(tmpdir) / 'tokens').mkdir()
    for i in range(32):
        torch.save({'acoustic_tokens': torch.zeros(50, 12).long(),
                    'semantic_tokens': torch.zeros(50).long(),
                    'pitch_tokens': torch.zeros(50).long(),
                    'speaker_id': i % 5},
                   Path(tmpdir) / 'tokens' / f's{i}.pt')
    ds = CoMelSingerDataset(tmpdir)
    sampler = BalancedSpeakerSampler(ds, batch_size=8)
    ok_count = 0
    for batch in sampler:
        spk_ids = [ds[i]['speaker_id'] for i in batch[:8]]
        if len(spk_ids) > len(set(spk_ids)):
            ok_count += 1
    ratio = ok_count / len(sampler)
    assert ratio == 1.0, f'pair guarantee failed: {ratio:.2f}'
    print(f'PASS: 100% batches have speaker pair ({ok_count}/{len(sampler)})')
"
```

## 5. 懸念事項とレビュー項目

- **話者が 1 人だけのデータセット**: `valid_speakers` が空になり `random.choice` が失敗する。`__init__` でアサーションを追加すること。
- **バッチサイズ > データ数**: `drop_last=True` でも小規模データで問題が起きる。デバッグ用途では `drop_last=False` を用意する。

### レビュー項目

- [ ] バッチ先頭 2 インデックスが同一話者か
- [ ] `valid_speakers` が 2 件以上サンプルを持つ話者のみか
- [ ] `__len__` が正しいバッチ数を返すか
- [ ] 全話者に 2 件以上サンプルがない場合に例外を送出するか

## 6. フェーズ振り返り: 一から作り直すとしたら

**torch Dataset vs WebDataset vs HF datasets**: WebDataset はシャードベースでサンプラーが異なる。WebDataset 採用の場合は `BalancedSpeakerSampler` は使えず、シャード内でのバランシングが必要になる。

**サンプリング戦略**: 現在の実装はバッチごとに 1 ペアだけ保証する。論文の SCL では全バッチペアを対象とするため、`k_s` 個全員が同一話者というより強い保証も検討する。

**分散学習対応**: `DistributedSampler` と組み合わせる場合、各ランクが独立したバッチを生成するため、ペア保証がランクをまたがない点に注意。

## 7. 後続タスクへの連絡事項

- **M2-09**: `collate_fn` は `BalancedSpeakerSampler` から返されるインデックスリストに対応した可変長パディングを実装すること。
- **M2-10**: `DataLoader(batch_sampler=sampler)` として使用するため、`batch_size` 引数を `DataLoader` に渡さないこと（二重指定エラーになる）。
- **M3（S2A 学習）**: SCL の positive/negative ペアは先頭 2 インデックスがペアであることを前提としている。`collate_fn` がペア位置を保持していることを確認すること。
