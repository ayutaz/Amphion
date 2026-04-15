# M3-02: DataLoader検証

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-06
> **依存チケット**: M2-10
> **ブロックするチケット**: M3-04
> **状態**: TODO

---

## 1. 目的とゴール

M2-10で実装したDatasetとBalancedSpeakerSamplerを組み合わせたDataLoaderを起動し、バッチのキー・shape・dtypeが学習コードの期待値と一致することをスクリプトで検証する。`acoustic_tokens`, `pitch_tokens`, `semantic_tokens`, `speaker_id` の4キーが正しく存在し、型とshapeが仕様通りであることを確認する。

## 2. 実装する内容の詳細

```python
# tools/verify_dataloader.py
import torch
from torch.utils.data import DataLoader
from models.tts.comelsinger.dataset import CoMelSingerDataset
from models.tts.comelsinger.sampler import BalancedSpeakerSampler

def verify_batch(batch, batch_size=4):
    expected_keys = {"acoustic_tokens", "pitch_tokens", "semantic_tokens", "speaker_id"}
    assert set(batch.keys()) == expected_keys, f"Missing keys: {expected_keys - set(batch.keys())}"

    B, T, Q = batch["acoustic_tokens"].shape
    assert Q == 8, f"Expected 8 codebooks, got {Q}"
    assert batch["acoustic_tokens"].dtype == torch.long
    assert batch["pitch_tokens"].dtype == torch.long
    assert batch["speaker_id"].shape == (B,)
    print(f"PASS: batch keys/shapes/dtypes OK (B={B}, T={T}, Q={Q})")
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 検証スクリプト担当 | 1 | verify_dataloader.py の実装・実行・結果確認 |

## 4. 提供範囲とテスト項目

**含むもの**: `tools/verify_dataloader.py` 検証スクリプト、BalancedSpeakerSampler動作確認

**含まないもの**: DatasetクラスやSamplerの実装変更（→ M2-10の責務）

### ユニットテスト

```bash
uv run python -c "
from models.tts.comelsinger.sampler import BalancedSpeakerSampler
# Samplerがイテラブルであること確認
s = BalancedSpeakerSampler(speaker_ids=[0,0,1,1,2,2], samples_per_speaker=2)
indices = list(iter(s))
assert len(indices) == 6
print('PASS: BalancedSpeakerSampler iteration OK')
"
```

```bash
uv run python -c "
import torch
batch = {
    'acoustic_tokens': torch.zeros(4, 100, 8, dtype=torch.long),
    'pitch_tokens': torch.zeros(4, 100, dtype=torch.long),
    'semantic_tokens': torch.zeros(4, 50, dtype=torch.long),
    'speaker_id': torch.zeros(4, dtype=torch.long),
}
from tools.verify_dataloader import verify_batch
verify_batch(batch)
"
```

### E2Eテスト

```bash
uv run python tools/verify_dataloader.py \
    --data_dir data/m4singer_processed \
    --batch_size 4 \
    --num_batches 5
# 5バッチ全てPASSすること
```

## 5. 懸念事項とレビュー項目

- **可変長シーケンスのパディング**: バッチ内でシーケンス長が異なる場合のcollate_fnの挙動を確認する。
- **話者バランス**: BalancedSpeakerSamplerが実際に各話者から均等にサンプリングしているか統計的に確認する。

### レビュー項目

- [ ] 4つのキーが全バッチで存在すること
- [ ] acoustic_tokensのcodebook数がQ=8であること
- [ ] dtypeがすべてtorch.longであること

## 6. フェーズ振り返り: 一から作り直すとしたら

- **PyTorch Lightning**: `LightningDataModule` を採用すれば、train/val/testのDataLoaderを統一インターフェースで管理でき、検証スクリプトが不要になる。
- **実験管理**: W&BやMLflowでデータ統計（話者分布、シーケンス長分布）を自動記録する仕組みを入れておく。
- **型アノテーション付きデータクラス**: バッチ構造を `TypedDict` や dataclass で定義し、静的型チェックで不整合を早期発見する。

## 7. 後続タスクへの連絡事項

- **M3-04**: 学習ループでDataLoaderを使用する際は、本チケットで確認した4キー構造を前提とすること。
- **M3-08**: S2A学習では同一バッチからpromptを生成するため、`speaker_id`キーが必須であることを再確認する。
- **M3-09**: バッチB(K=32)をS_A_sとS_A_fに分割する際、acoustic_tokensのshapeが (32, T, 8) であることを前提とする。
