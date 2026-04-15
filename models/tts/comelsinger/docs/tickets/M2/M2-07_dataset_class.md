# M2-07: CoMelSingerDataset クラス定義

> **マイルストーン**: [M2: コアモジュール](../../13_milestones.md#m2-コアモジュール実装)
> **対応RQ**: RQ-06
> **依存チケット**: M1
> **ブロックするチケット**: M2-08
> **状態**: TODO

---

## 1. 目的とゴール

`dataset.py` に `CoMelSingerDataset(torch.utils.data.Dataset)` を定義する。前処理済みの `tokens/{uid}.pt` を読み込み、`acoustic_tokens` を `(12, T)` 形状で返す。`speaker_id` も辞書に含め、M2-08 の `BalancedSpeakerSampler` が使えるようにする。

このクラスが M2-08〜M2-10 のデータパイプライン全体の基盤となる。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/dataset.py
from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import Dataset

class CoMelSingerDataset(Dataset):
    """前処理済みトークンを読み込む Dataset。
    tokens/{uid}.pt: {
        "acoustic_tokens": (12, T) または (T, 12),
        "semantic_tokens": (T,),
        "pitch_tokens":    (T,),
        "speaker_id":      int,
        "uid":             str,
    }
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.token_dir = self.data_dir / "tokens"
        self.samples: List[Path] = sorted(self.token_dir.glob("*.pt"))
        if len(self.samples) == 0:
            raise ValueError(f"No .pt files found in {self.token_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = torch.load(self.samples[idx], weights_only=True)
        acoustic = data["acoustic_tokens"]
        # (T, 12) → (12, T) に統一
        if acoustic.shape[-1] == 12:
            acoustic = acoustic.T
        return {
            "acoustic_tokens": acoustic,          # (12, T)
            "semantic_tokens": data["semantic_tokens"],  # (T,)
            "pitch_tokens":    data["pitch_tokens"],     # (T,)
            "speaker_id":      int(data["speaker_id"]),
            "uid":             str(data.get("uid", self.samples[idx].stem)),
        }
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `dataset.py` の新規作成、`CoMelSingerDataset` の実装 |

## 4. 提供範囲とテスト項目

**含むもの**: `dataset.py` の新規作成、`CoMelSingerDataset` の `__init__` / `__len__` / `__getitem__` の実装

**含まないもの**: `BalancedSpeakerSampler`（→ M2-08）、`collate_fn`（→ M2-09）

### ユニットテスト

```bash
uv run python -c "
import torch, tempfile
from pathlib import Path
from models.tts.comelsinger.dataset import CoMelSingerDataset
# dummy データ作成
with tempfile.TemporaryDirectory() as tmpdir:
    (Path(tmpdir) / 'tokens').mkdir()
    torch.save({
        'acoustic_tokens': torch.randint(0, 1024, (100, 12)),
        'semantic_tokens': torch.randint(0, 1024, (100,)),
        'pitch_tokens':    torch.randint(0,  129, (100,)),
        'speaker_id': 0,
    }, Path(tmpdir) / 'tokens' / 'sample_0.pt')
    ds = CoMelSingerDataset(tmpdir)
    assert len(ds) == 1
    item = ds[0]
    assert item['acoustic_tokens'].shape[0] == 12
    print('PASS: __getitem__ shape OK')
"
```

```bash
uv run python -c "
import torch, tempfile
from pathlib import Path
from models.tts.comelsinger.dataset import CoMelSingerDataset
with tempfile.TemporaryDirectory() as tmpdir:
    (Path(tmpdir) / 'tokens').mkdir()
    torch.save({'acoustic_tokens': torch.zeros(50, 12).long(),
                'semantic_tokens': torch.zeros(50).long(),
                'pitch_tokens': torch.zeros(50).long(),
                'speaker_id': 1}, Path(tmpdir) / 'tokens' / 'a.pt')
    ds = CoMelSingerDataset(tmpdir)
    assert isinstance(ds[0]['speaker_id'], int)
    print('PASS: speaker_id is int OK')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch, tempfile
from pathlib import Path
from models.tts.comelsinger.dataset import CoMelSingerDataset
with tempfile.TemporaryDirectory() as tmpdir:
    (Path(tmpdir) / 'tokens').mkdir()
    for i in range(5):
        torch.save({'acoustic_tokens': torch.randint(0, 1024, (80, 12)),
                    'semantic_tokens': torch.randint(0, 1024, (80,)),
                    'pitch_tokens':    torch.randint(0, 129,  (80,)),
                    'speaker_id': i % 2}, Path(tmpdir) / 'tokens' / f's{i}.pt')
    ds = CoMelSingerDataset(tmpdir)
    assert len(ds) == 5
    for item in ds:
        assert item['acoustic_tokens'].shape[0] == 12
    print('PASS: all items shape OK')
"
```

## 5. 懸念事項とレビュー項目

- **`weights_only=True`**: PyTorch 2.x の推奨設定。旧フォーマットのデータとの互換性を確認すること。
- **大規模データ**: M4Singer（≈29時間）は数万ファイルになる可能性がある。`glob("*.pt")` のソートがメモリ上で行われるため、100k ファイル超では事前にインデックスファイル（`index.json`）を生成する方式を検討する。

### レビュー項目

- [ ] `acoustic_tokens.shape[0] == 12` か（12 codebooks が第0次元）
- [ ] `speaker_id` が `int` 型か
- [ ] 空ディレクトリで `ValueError` が送出されるか
- [ ] `(T, 12)` と `(12, T)` の両入力形式に対応しているか

## 6. フェーズ振り返り: 一から作り直すとしたら

**torch Dataset vs WebDataset vs HF datasets**: 大規模データセット（>100GB）では WebDataset（shard ベース streaming）が有利。今回の ≈35 時間規模なら torch Dataset で十分だが、将来拡張を考えると `__init__` に `streaming: bool = False` オプションを持たせる。

**サンプリング戦略**: 現在は `glob` によるファイルリストだが、話者ラベルが必要な `BalancedSpeakerSampler`（M2-08）のために `speaker_id → [indices]` の逆引き辞書を `__init__` で構築しておくと効率的。

**HF datasets との互換**: `__getitem__` の戻り値を HuggingFace `datasets` ライブラリのカラム形式と互換にしておくと、将来的な移行が容易。

## 7. 後続タスクへの連絡事項

- **M2-08**: `CoMelSingerDataset.__init__` で `{speaker_id: [indices]}` 形式の `self.speaker_index` 辞書を構築しておくと `BalancedSpeakerSampler` の実装が容易。
- **M2-09**: `__getitem__` の戻り値の shape と dtype を `collate_fn` 実装前に文書化すること。
- **M2-10**: `DataLoader` の `num_workers > 0` では `torch.load` が各ワーカーで独立して実行される。ファイルディスクリプタの枯渇に注意。
