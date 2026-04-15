# M2-10: DataLoader 統合動作確認

> **マイルストーン**: [M2: コアモジュール](../../13_milestones.md#m2-コアモジュール実装)
> **対応RQ**: RQ-06
> **依存チケット**: M2-09
> **ブロックするチケット**: M3-02
> **状態**: TODO

---

## 1. 目的とゴール

`CoMelSingerDataset`、`BalancedSpeakerSampler`、`comelsinger_collate_fn` を組み合わせた `DataLoader` で 1 バッチを取得し、全テンソルの形状が仕様通りであることを確認する。取得は 60 秒以内に完了すること。

このチケットは M2 データパイプラインの最終統合確認であり、M3 以降の学習パイプラインが依存する。

## 2. 実装する内容の詳細

```python
# tests/test_dataloader_integration.py
import time, tempfile, torch
from pathlib import Path
from torch.utils.data import DataLoader
from models.tts.comelsinger.dataset import (
    CoMelSingerDataset,
    BalancedSpeakerSampler,
    comelsinger_collate_fn,
)

def create_dummy_dataset(tmpdir: str, n: int = 32, n_spk: int = 4) -> str:
    token_dir = Path(tmpdir) / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        T = 60 + (i % 40)   # 60-99 フレームの可変長
        torch.save({
            "acoustic_tokens": torch.randint(0, 1024, (T, 12)),
            "semantic_tokens": torch.randint(0, 1024, (T,)),
            "pitch_tokens":    torch.randint(0, 129,  (T,)),
            "speaker_id": i % n_spk,
        }, token_dir / f"sample_{i:04d}.pt")
    return tmpdir

def test_dataloader_one_batch():
    BATCH_SIZE = 8
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir)
        ds      = CoMelSingerDataset(tmpdir)
        sampler = BalancedSpeakerSampler(ds, batch_size=BATCH_SIZE)
        loader  = DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=comelsinger_collate_fn,
            num_workers=2,
        )
        start = time.time()
        batch = next(iter(loader))
        elapsed = time.time() - start
        assert elapsed < 60, f"DataLoader took {elapsed:.1f}s (>60s)"
        T_max = batch["acoustic_tokens"].shape[-1]
        assert batch["acoustic_tokens"].shape  == (BATCH_SIZE, 12, T_max)
        assert batch["semantic_tokens"].shape  == (BATCH_SIZE, T_max)
        assert batch["pitch_tokens"].shape     == (BATCH_SIZE, T_max)
        assert batch["attention_mask"].shape   == (BATCH_SIZE, T_max)
        assert batch["speaker_ids"].shape      == (BATCH_SIZE,)
        print(f"PASS: 1 batch in {elapsed:.2f}s, T_max={T_max}")

if __name__ == "__main__":
    test_dataloader_one_batch()
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| テストエージェント | 1 | 統合テストスクリプトの作成・実行・問題フィードバック |

## 4. 提供範囲とテスト項目

**含むもの**: 統合テストスクリプト `tests/test_dataloader_integration.py` の作成と実行

**含まないもの**: DataLoader の実装変更（問題があれば M2-07〜M2-09 へフィードバック）

### ユニットテスト

```bash
uv run python -c "
import torch, tempfile
from pathlib import Path
from torch.utils.data import DataLoader
from models.tts.comelsinger.dataset import CoMelSingerDataset, BalancedSpeakerSampler, comelsinger_collate_fn
with tempfile.TemporaryDirectory() as tmpdir:
    (Path(tmpdir) / 'tokens').mkdir()
    for i in range(16):
        torch.save({'acoustic_tokens': torch.zeros(50+i,12).long(),
                    'semantic_tokens': torch.zeros(50+i).long(),
                    'pitch_tokens': torch.zeros(50+i).long(),
                    'speaker_id': i%3}, Path(tmpdir)/'tokens'/f's{i}.pt')
    ds = CoMelSingerDataset(tmpdir)
    loader = DataLoader(ds, batch_sampler=BalancedSpeakerSampler(ds,8), collate_fn=comelsinger_collate_fn)
    batch = next(iter(loader))
    assert batch['acoustic_tokens'].shape[0] == 8
    print('PASS: DataLoader batch_size=8 OK')
"
```

```bash
uv run python -c "
import torch, tempfile
from pathlib import Path
from torch.utils.data import DataLoader
from models.tts.comelsinger.dataset import CoMelSingerDataset, BalancedSpeakerSampler, comelsinger_collate_fn
with tempfile.TemporaryDirectory() as tmpdir:
    (Path(tmpdir) / 'tokens').mkdir()
    for i in range(16):
        torch.save({'acoustic_tokens': torch.zeros(50,12).long(),
                    'semantic_tokens': torch.zeros(50).long(),
                    'pitch_tokens': torch.zeros(50).long(),
                    'speaker_id': i%4}, Path(tmpdir)/'tokens'/f's{i}.pt')
    ds = CoMelSingerDataset(tmpdir)
    loader = DataLoader(ds, batch_sampler=BalancedSpeakerSampler(ds,4), collate_fn=comelsinger_collate_fn, num_workers=2)
    batch = next(iter(loader))
    print('PASS: num_workers=2 OK')
"
```

### E2Eテスト

```bash
uv run python tests/test_dataloader_integration.py
# 期待出力: PASS: 1 batch in X.XXs, T_max=XX
```

## 5. 懸念事項とレビュー項目

- **num_workers とマルチプロセス**: macOS では `num_workers > 0` が `fork` の代わりに `spawn` を使うため、初期化コストが高い。CI 環境では `num_workers=0` で実行すること。
- **60 秒制限**: 実データ（M4Singer）での計測も M3 開始前に行うこと。

### レビュー項目

- [ ] 1 バッチ取得が 60 秒以内か
- [ ] `acoustic_tokens.shape == (B, 12, T_max)` か
- [ ] `attention_mask` の形状が正しいか
- [ ] `num_workers=2` でデッドロックが起きないか

## 6. フェーズ振り返り: 一から作り直すとしたら

**torch Dataset vs WebDataset vs HF datasets**: WebDataset なら `wds.WebLoader` を使い、シャードをストリーミングするため 60 秒制限の懸念がない。今回規模では不要だが、将来拡張を見据えた設計。

**サンプリング戦略**: バケットサンプリング（同程度の長さをバッチ化）を加えると T_max が小さくなり GPU メモリ効率が向上する。`BalancedSpeakerSampler` との組み合わせが複雑になるため、別クラスとして実装する。

**分散学習対応**: `DistributedSampler` と `BalancedSpeakerSampler` を組み合わせる設計は非自明。各ランクが独立したバランシングを行う `DistributedBalancedSpeakerSampler` が必要になる。

## 7. 後続タスクへの連絡事項

- **M3-02（学習ループ）**: `DataLoader` のイテレーションは本テストで動作確認済み。学習ループでは `for batch in loader:` の形式で使用すること。
- **M3（S2A 学習）**: `batch["attention_mask"]` は `(B, T_max)` FloatTensor（1=有効）。MaskGCT の `x_mask` と同じ形式か確認すること。
- **M3（分散学習）**: `DistributedSampler` への切り替えが必要な場合は `BalancedSpeakerSampler` を修正すること。
