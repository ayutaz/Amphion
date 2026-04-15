# M1-12: 音響トークン抽出 (extract_acoustic_tokens)

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-05
> **依存チケット**: M0-01, M0-10
> **ブロックするチケット**: M1-16, M1-18
> **状態**: TODO

---

## 1. 目的とゴール

`models/tts/comelsinger/preprocess.py` を新規作成し、Amphion Codec を用いて音声波形から RVQ 12層の音響トークン列を抽出する関数を実装する。

**ゴール**: 入力波形 (24kHz) から `acoustic_tokens (T_a, 12)` long型、値域 `[0, 1023]` を出力。

## 2. 実装する内容の詳細

```python
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
import torch

def extract_acoustic_tokens(
    speech: torch.Tensor,          # (T,), 24kHz, float32
    codec_encoder: CodecEncoder,
    codec_decoder: CodecDecoder,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    speech = speech.to(device)
    with torch.no_grad():
        vq_emb = codec_encoder(speech.unsqueeze(0).unsqueeze(0))
        _, vq, _, _, _ = codec_decoder.quantizer(vq_emb)
        acoustic_tokens = vq.permute(1, 2, 0).squeeze(0).long()  # (T_a, 12)
    return acoustic_tokens
```

**注意**: `maskgct_utils.py` のパターンに準拠。codec_encoder/decoder は呼び出し元でロード済みインスタンスを渡す。

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当 |
|---|---|---|
| 実装 | 1 | preprocess.py 新規作成・関数実装 |
| レビュー | 1 | Codec API・quantizer 返り値順序の確認 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲

`models/tts/comelsinger/preprocess.py` に `extract_acoustic_tokens` を実装

### 4.2 ユニットテスト

- shape `(T_a, 12)`、dtype `torch.long`、値域 `[0, 1023]` を確認
- 同一音声で冪等であること（2回実行で同一結果）

### 4.3 E2Eテスト

- M1-18 統合テストで合成正弦波を通じて確認（`@pytest.mark.slow`）

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- `codec_decoder.quantizer` 返り値の順序（`_, vq, _, _, _`）がバージョンで異なる可能性
- 長音声（> 60秒）で GPU OOM の可能性 → チャンク分割を検討

### 5.2 レビュー項目

- [ ] `vq.permute(1, 2, 0)` の次元変換が正しいか
- [ ] RVQ 層数が 12 であることを `assert` で検証しているか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **Codec バックエンド抽象化**: `CodecBackend` プロトコルで EnCodec/Amphion Codec を統一インターフェースで切替可能に
- **ストリーミング処理**: 長音声対応のフレーム単位ストリーミング抽出
- **データフォーマット**: `.pt` 個別保存より HDF5/WebDataset の方がランダムアクセス効率が高い

## 7. 後続タスクへの連絡事項

- **M1-16**: 保存時に `acoustic_tokens` を `(12, T_a)` に転置する（DataLoader の入力形式に合わせる）
- **M1-13, M1-14**: `T_a`（音響トークン長）を `target_len` として渡し、フレーム数を統一すること
