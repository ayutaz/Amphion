# M1-05: decode_token_to_freq 実装

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-01
> **依存チケット**: M1-01
> **ブロックするチケット**: M1-06
> **状態**: TODO

---

## 1. 目的とゴール

`PitchTokenizer.decode_token_to_freq()` を実装する。離散ピッチトークン（整数）を受け取り、対応する周波数（Hz）を返す。これは `quantize_f0` の逆変換であり、評価時に推論結果の離散ピッチトークンを連続周波数値に変換するために使用する。

## 2. 実装する内容の詳細

### 2.1 変換式

```
token = 0  → 0.0 Hz (unvoiced)
token = n  → 440 * 2^((n-69)/12)  Hz  (MIDI → Hz)
```

MIDI 69 が A4=440Hz の基準音。`n-69` が半音数のオフセット。

### 2.2 実装コード

```python
def decode_token_to_freq(self, token: int) -> float:
    """
    ピッチトークン → Hz 逆変換

    Args:
        token: ピッチトークン, int, 値域 [0, 128]
               0: 無声 → 0.0 Hz
               1-128: MIDI番号 → 対応周波数 Hz

    Returns:
        freq: 周波数 Hz, float
              token=0 → 0.0
              token=n → 440 * 2^((n-69)/12)
    """
    if token == 0:
        return 0.0
    return 440.0 * (2.0 ** ((token - 69) / 12.0))
```

### 2.3 代表値テーブル

| Token (MIDI) | 音名 | 周波数 (Hz) |
|---|---|---|
| 0 | 無声 | 0.0 |
| 57 | A3 | 220.0 |
| 60 | C4 (Middle C) | 261.63 |
| 69 | A4 | 440.0 |
| 72 | C5 | 523.25 |
| 81 | A5 | 880.0 |
| 128 | G#9/Ab9 | 12543.9 |

### 2.4 精度に関する注意

`quantize_f0` では `round()` で離散化するため、逆変換後の周波数は元の F0 と厳密には一致しない。例えば、441Hz を量子化すると MIDI 69 になり、逆変換で 440Hz が返る（約0.2%の誤差）。

これは離散化の本質的な誤差であり、評価指標（F0-RMSE）は量子化後のトークン空間での比較を行うため問題ない。

**重要: Python は全て `uv run python` / `uv run pytest` で実行。pip 禁止。**

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `decode_token_to_freq` メソッドの実装 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: `decode_token_to_freq` メソッドの完全実装

**含まないもの**: テストファイル作成（→ M1-06）、バッチ変換ユーティリティ（評価スクリプト側で実装）

### 4.2 ユニットテスト

```bash
uv run python -c "
import math
from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer
tok = PitchTokenizer()

# 無声 → 0.0
assert tok.decode_token_to_freq(0) == 0.0
print('PASS: token 0 -> 0.0 Hz')

# A4 (MIDI 69) → 440.0
result = tok.decode_token_to_freq(69)
assert abs(result - 440.0) < 1e-6, f'expected 440.0, got {result}'
print('PASS: token 69 -> 440.0 Hz')

# A3 (MIDI 57) → 220.0
result = tok.decode_token_to_freq(57)
assert abs(result - 220.0) < 1e-6, f'expected 220.0, got {result}'
print('PASS: token 57 -> 220.0 Hz')

# A5 (MIDI 81) → 880.0
result = tok.decode_token_to_freq(81)
assert abs(result - 880.0) < 1e-6, f'expected 880.0, got {result}'
print('PASS: token 81 -> 880.0 Hz')

# 全トークン範囲で非負・有限値を確認
for t in range(0, 129):
    f = tok.decode_token_to_freq(t)
    assert f >= 0.0, f'negative freq at token {t}'
    assert math.isfinite(f), f'non-finite at token {t}'
print('PASS: all tokens [0,128] return non-negative finite values')
"
```

### 4.3 E2Eテスト（quantize_f0 との往復検証）

```bash
uv run python -c "
import numpy as np, math
from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer
tok = PitchTokenizer()

# quantize_f0 → decode_token_to_freq の往復で元の周波数に近い値が返る
test_freqs = [220.0, 261.63, 440.0, 523.25, 880.0]
for f_orig in test_freqs:
    tokens = tok.quantize_f0(np.array([f_orig] * 3), 1)
    f_decoded = tok.decode_token_to_freq(tokens[0].item())
    # 量子化誤差は1セント（約0.06%）以内
    if f_orig > 0:
        cents_error = abs(1200 * math.log2(f_decoded / f_orig))
        assert cents_error < 60, f'{f_orig}Hz: {cents_error:.1f} cents error'
    print(f'PASS: {f_orig:.1f}Hz -> token {tokens[0].item()} -> {f_decoded:.1f}Hz')
"
```

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- **精度**: `float` (64-bit) で計算するため、実用上の精度は十分。ただし GPU上での大規模計算で `torch.float32` を使用する場合は精度が落ちる。評価スクリプトでは `float64` を使うことを推奨。
- **境界値**: token=128 の場合、`440 * 2^((128-69)/12) ≈ 12543.9 Hz`。これは可聴域上限（約20kHz）以内であり問題ない。

### 5.2 レビュー項目

- [ ] `decode_token_to_freq(69)` が `440.0` Hz に近い値を返すか（誤差 < 1e-6）
- [ ] `decode_token_to_freq(0)` が `0.0` を返すか
- [ ] 全トークン範囲 [0, 128] で非負・有限値が返るか

## 6. フェーズ振り返り: 一から作り直すとしたら

**量子化誤差の定量化**

`quantize_f0` と `decode_token_to_freq` の往復誤差（量子化誤差）を `__init__` 段階で事前計算し、`self.quantization_error_cents` として格納しておくとデバッグが容易になる。平均量子化誤差は約 30 セント（半音の 30%）程度。

**ベクトル化**

単一 `int` を受け取るインターフェースは評価スクリプトでループが必要になる。一から設計するなら `decode_tokens_to_freqs(tokens: torch.LongTensor) -> torch.FloatTensor` のベクトル化版を提供し、単一版は内部でベクトル版を呼ぶ設計にする。

**型安全性**

現在の実装は `token: int` だが、`torch.Tensor` が渡された場合も処理できるように `int(token)` で変換するか、型チェックを追加することを推奨する。

## 7. 後続タスクへの連絡事項

- **M1-06 (test_pitch_tokenizer)**: `decode_token_to_freq(69) ≈ 440.0` を必ずテストに含めること。往復テスト（`quantize_f0 → decode_token_to_freq`）も推奨。
- **M4 評価スクリプト**: 推論結果の `(T_a,)` ピッチトークン列を周波数列に変換する際に `decode_token_to_freq` を使用する。バッチ処理のため、リスト内包表記または `torch.tensor([decode_token_to_freq(t.item()) for t in tokens])` パターンを使用する。
- **M2〜M4**: このメソッドは評価時にのみ使用する。学習・推論の主パスでは使用しない。
