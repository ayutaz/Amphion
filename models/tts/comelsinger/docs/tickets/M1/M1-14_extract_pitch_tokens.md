# M1-14: F0抽出とピッチトークン化 (extract_pitch_tokens)

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-05
> **依存チケット**: M1-02, M0-05
> **ブロックするチケット**: M1-16, M1-18
> **状態**: TODO

---

## 1. 目的とゴール

pyworld DIO+StoneMask で F0 を抽出し、`PitchTokenizer.quantize_f0` で離散ピッチトークンに変換する。

**ゴール**: `pitch_tokens (T_a,)` long型、値域 `[0, 128]`、75Hz 基準。

## 2. 実装する内容の詳細

```python
import numpy as np
import pyworld as pw

def extract_pitch_tokens(
    speech: torch.Tensor,           # (T,), 24kHz, float32
    pitch_tokenizer,                # PitchTokenizer
    target_len: int,                # T_a (75Hz)
    sr: int = 24000,
    frame_period: float = 5.0,      # ms → 200Hz
    f0_floor: float = 65.0,
    f0_ceil: float = 1047.0,        # C6
) -> torch.Tensor:
    speech_np = speech.numpy().astype(np.float64)  # pyworld は float64 必須
    f0, t = pw.dio(speech_np, sr, f0_floor=f0_floor, f0_ceil=f0_ceil, frame_period=frame_period)
    f0 = pw.stonemask(speech_np, f0, t, sr)  # refinement
    pitch_tokens = pitch_tokenizer.quantize_f0(f0, target_len=target_len)
    return pitch_tokens  # (T_a,)
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当 |
|---|---|---|
| 実装 | 1 | 関数実装 |
| レビュー | 1 | pyworld パラメータ確認 |

## 4. 提供範囲とテスト項目

### 4.2 ユニットテスト

- 440Hz 正弦波 → MIDI 69 (A4) 付近のトークンが80%以上
- shape `(T_a,)`、dtype `torch.long`、値域 `[0, 128]`

### 4.3 E2Eテスト

- M1-18 統合テストで M1-12 の `T_a` を `target_len` に渡し長さ整合を確認

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- pyworld は `float64` numpy 配列を要求。`float32` だと無音扱いされる
- `f0_floor=65Hz` が歌唱音声に適切か論文設定を再確認
- `stonemask` の第3引数 `t` を正しく渡すこと

### 5.2 レビュー項目

- [ ] `speech.numpy().astype(np.float64)` で型変換しているか
- [ ] `quantize_f0` の `target_len` が音響トークン長 `T_a` と一致しているか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **CREPE ベース F0**: pyworld より精度が高い。GPU 依存だが学習品質に直結
- **連続値 F0 保存**: raw F0 を保存し離散化を遅延実行。量子化解像度を後から変更可能
- **フレームレート統一**: 200Hz F0 → 75Hz 変換を共通ユーティリティとして切り出す

## 7. 後続タスクへの連絡事項

- **M1-16**: `pitch_tokens` を `(T_a,)` shape で保存
- **M2 (SVT)**: SVT モジュールのターゲットラベルとしてこのトークン化結果を使用
