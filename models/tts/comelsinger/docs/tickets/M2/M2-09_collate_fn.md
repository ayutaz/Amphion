# M2-09: comelsinger_collate_fn 実装

> **マイルストーン**: [M2: コアモジュール](../../13_milestones.md#m2-コアモジュール実装)
> **対応RQ**: RQ-06
> **依存チケット**: M2-08
> **ブロックするチケット**: M2-10
> **状態**: TODO

---

## 1. 目的とゴール

`dataset.py` に `comelsinger_collate_fn` を実装する。可変長の音響・セマンティック・ピッチトークンをゼロパディングし、`acoustic_tokens: (B, 12, T_max)`、`semantic_tokens: (B, T_max)`、`pitch_tokens: (B, T_max)`、`attention_mask: (B, T_max)`（1=有効, 0=パディング）を返す。

`attention_mask` は下流の損失計算とマスク生成で共通利用するため、padding_mask（True=パディング）と同じ情報を逆符号で持つ。

## 2. 実装する内容の詳細

```python
# dataset.py に追加
from typing import List, Dict, Any
import torch

def comelsinger_collate_fn(
    batch: List[Dict[str, Any]]
) -> Dict[str, torch.Tensor]:
    """可変長トークン列をパディングしてバッチ化する。

    Returns:
        acoustic_tokens:  (B, 12, T_max) LongTensor
        semantic_tokens:  (B, T_max)     LongTensor
        pitch_tokens:     (B, T_max)     LongTensor
        attention_mask:   (B, T_max)     FloatTensor (1=valid, 0=pad)
        speaker_ids:      (B,)           LongTensor
    """
    B = len(batch)
    lengths = [item["acoustic_tokens"].shape[-1] for item in batch]
    T_max = max(lengths)

    acoustic = torch.zeros(B, 12, T_max, dtype=torch.long)
    semantic = torch.zeros(B, T_max, dtype=torch.long)
    pitch    = torch.zeros(B, T_max, dtype=torch.long)
    mask     = torch.zeros(B, T_max, dtype=torch.float)
    spk_ids  = torch.zeros(B, dtype=torch.long)

    for i, (item, T) in enumerate(zip(batch, lengths)):
        acoustic[i, :, :T] = item["acoustic_tokens"]   # (12, T)
        semantic[i, :T]    = item["semantic_tokens"]    # (T,)
        pitch[i, :T]       = item["pitch_tokens"]       # (T,)
        mask[i, :T]        = 1.0
        spk_ids[i]         = item["speaker_id"]

    return {
        "acoustic_tokens": acoustic,
        "semantic_tokens": semantic,
        "pitch_tokens":    pitch,
        "attention_mask":  mask,
        "speaker_ids":     spk_ids,
    }
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `comelsinger_collate_fn` の実装 |

## 4. 提供範囲とテスト項目

**含むもの**: `comelsinger_collate_fn` 関数の実装

**含まないもの**: DataLoader への組み込み（→ M2-10）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.dataset import comelsinger_collate_fn
batch = [
    {'acoustic_tokens': torch.randint(0,1024,(12,80)), 'semantic_tokens': torch.randint(0,1024,(80,)),
     'pitch_tokens': torch.randint(0,129,(80,)), 'speaker_id': 0},
    {'acoustic_tokens': torch.randint(0,1024,(12,50)), 'semantic_tokens': torch.randint(0,1024,(50,)),
     'pitch_tokens': torch.randint(0,129,(50,)), 'speaker_id': 1},
]
out = comelsinger_collate_fn(batch)
assert out['acoustic_tokens'].shape == (2, 12, 80)
assert out['semantic_tokens'].shape == (2, 80)
assert out['pitch_tokens'].shape    == (2, 80)
assert out['attention_mask'].shape  == (2, 80)
print('PASS: output shapes OK')
"
```

```bash
uv run python -c "
import torch
from models.tts.comelsinger.dataset import comelsinger_collate_fn
batch = [
    {'acoustic_tokens': torch.ones(12,60,dtype=torch.long),
     'semantic_tokens': torch.ones(60,dtype=torch.long),
     'pitch_tokens': torch.ones(60,dtype=torch.long), 'speaker_id': 0},
    {'acoustic_tokens': torch.ones(12,40,dtype=torch.long)*2,
     'semantic_tokens': torch.ones(40,dtype=torch.long)*2,
     'pitch_tokens': torch.ones(40,dtype=torch.long)*2, 'speaker_id': 1},
]
out = comelsinger_collate_fn(batch)
# パディング位置はゼロ
assert out['acoustic_tokens'][1, 0, 40:].sum() == 0, 'padding should be 0'
assert out['attention_mask'][1, 40:].sum()      == 0, 'mask padding should be 0'
assert out['attention_mask'][1, :40].sum()      == 40, 'mask valid should be 1.0'
print('PASS: padding OK')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.dataset import comelsinger_collate_fn
# バッチサイズ4の動作確認
batch = [{'acoustic_tokens': torch.randint(0,1024,(12,T)),
          'semantic_tokens': torch.randint(0,1024,(T,)),
          'pitch_tokens': torch.randint(0,129,(T,)),
          'speaker_id': i}
         for i, T in enumerate([80, 65, 90, 70])]
out = comelsinger_collate_fn(batch)
T_max = 90
assert out['acoustic_tokens'].shape == (4, 12, T_max)
assert out['attention_mask'].sum().item() == 80+65+90+70, 'mask sum should equal total valid frames'
print('PASS: batch=4 collate OK')
"
```

## 5. 懸念事項とレビュー項目

- **`acoustic_tokens` の形状**: `CoMelSingerDataset.__getitem__` が `(12, T)` を返す前提。`(T, 12)` が返ってくる場合は `collate_fn` 冒頭で転置が必要。
- **padding トークン値**: 現在はゼロパディングだが、ゼロは有効なトークン ID（Embedding の index 0）である。損失計算では `attention_mask` でマスクするため問題ないが、明示的に `PAD_ID = 0` として文書化すること。

### レビュー項目

- [ ] `acoustic_tokens.shape == (B, 12, T_max)` か
- [ ] `attention_mask` の有効フレーム数の合計が入力長の合計と一致するか
- [ ] パディング位置が 0 埋めか
- [ ] `speaker_ids` が `(B,)` の LongTensor か

## 6. フェーズ振り返り: 一から作り直すとしたら

**torch Dataset vs WebDataset vs HF datasets**: HF `datasets` の `DataCollatorWithPadding` は文字列キーと `torch.Tensor` を自動パディングできる。`comelsinger_collate_fn` を HF 互換にするには戻り値を `BatchEncoding` 形式にする。

**サンプリング戦略**: 長さの近いサンプルをバッチ化する「バケットサンプリング」を使うと、T_max が大きくなりすぎるパディング無駄が減る。`BalancedSpeakerSampler` と組み合わせる設計も検討する。

**padding値の統一**: `PAD_ID = 0` はコードブック範囲（0〜1023）と重なる。学習に問題はないが、`ACOUSTIC_PAD = -1` のように無効値を使うと `attention_mask` なしでもパディング位置を識別できる。

## 7. 後続タスクへの連絡事項

- **M2-10**: `DataLoader(collate_fn=comelsinger_collate_fn)` として渡す。`batch_sampler` との組み合わせで `batch_size` 引数は不要。
- **M3（S2A 学習ループ）**: `attention_mask` は `padding_mask`（`True=パディング`）に変換して Transformer に渡す必要がある（`~attention_mask.bool()`）。
- **M3（SVT 損失）**: `attention_mask` を使って `logits[valid]` 形式で有効フレームのみ損失計算すること（M2-06 参照）。
