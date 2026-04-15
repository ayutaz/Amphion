# M1-02: quantize_f0 実装

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-01
> **依存チケット**: M1-01
> **ブロックするチケット**: M1-06, M1-14
> **状態**: TODO

---

## 1. 目的とゴール

`PitchTokenizer.quantize_f0()` を実装する。pyworld DIO+StoneMask で得られた 200Hz F0 配列 `(T_f0,)` を受け取り、各 Amphion Codec フレーム（75Hz）に対応する区間の有声 F0 中央値を MIDI 番号に変換し、離散ピッチトークン列 `(L,)` を返す。

このチケットを完了すると、F0 抽出済み音声データを離散トークンに変換できるようになり、M1-14（前処理パイプラインでのF0抽出・量子化）が実装可能になる。

## 2. 実装する内容の詳細

### 2.1 アルゴリズム詳細

```
F0_FPS=200Hz, ENCODEC_FPS=75Hz のとき:
  - 各 EnCodec フレーム i (0 ≤ i < target_len) が対応する F0 インデックス範囲:
      start = round(i * F0_FPS / ENCODEC_FPS)
      end   = round((i+1) * F0_FPS / ENCODEC_FPS)
  - その区間の有声フレーム (f0 > 0) の中央値を取る
  - 有声フレームがなければ token = 0 (unvoiced)
  - MIDI変換: midi = round(12 * log2(f / 440) + 69)
  - クランプ: token = clamp(midi, 1, 128)
```

### 2.2 実装コード

```python
def quantize_f0(self, f0: np.ndarray, target_len: int) -> torch.LongTensor:
    """
    F0 (T_f0,) Hz → ピッチトークン (L,)

    Args:
        f0: F0 配列 (T_f0,)、単位 Hz。無声フレームは 0.0。
        target_len: 出力トークン列の長さ (= 音響トークンフレーム数 T_a)

    Returns:
        tokens: shape (target_len,), dtype torch.long, 値域 [0, 128]
    """
    tokens = torch.zeros(target_len, dtype=torch.long)

    for i in range(target_len):
        start = round(i * self.f0_fps / self.encodec_fps)
        end   = round((i + 1) * self.f0_fps / self.encodec_fps)
        end   = min(end, len(f0))
        frame = f0[start:end]

        voiced = frame[frame > 0]
        if len(voiced) == 0:
            tokens[i] = 0  # unvoiced
            continue

        f_median = float(np.median(voiced))
        midi = round(12 * math.log2(f_median / 440.0) + 69)
        tokens[i] = max(1, min(128, midi))

    return tokens
```

### 2.3 エッジケース処理

| 入力条件 | 処理 |
|---|---|
| `f0` 全て 0.0 | 全トークン 0 (全無声) |
| `f0` 全て有声 | 各フレーム中央値から MIDI 変換 |
| `target_len > len(f0) * ENCODEC_FPS/F0_FPS` | 末尾フレームは `end = min(end, len(f0))` でクリップ |
| 極端に高い/低い F0 | `clamp(1, 128)` で MIDI 1〜128 に収める |
| `f0` 長さが 1 の場合 | 正常動作（区間 [0, 1) 内で処理） |

**重要: Python は全て `uv run python` / `uv run pytest` で実行。pip 禁止。**

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `quantize_f0` メソッドの実装 |
| レビューエージェント | 1 | 境界値・数値精度レビュー |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: `quantize_f0` メソッドの完全実装（M1-01 の `NotImplementedError` を置き換え）

**含まないもの**: テストファイル作成（→ M1-06）、F0 抽出自体（→ M1-14）

### 4.2 ユニットテスト

```bash
uv run python -c "
import numpy as np
from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer
tok = PitchTokenizer()

# 基本ケース: 440Hz → MIDI 69
result = tok.quantize_f0(np.array([440.0] * 3), 1)
assert result[0].item() == 69, f'expected 69, got {result[0].item()}'
print('PASS: 440Hz -> 69')

# 無声: 0.0 → token 0
result = tok.quantize_f0(np.array([0.0] * 3), 1)
assert result[0].item() == 0, f'expected 0, got {result[0].item()}'
print('PASS: unvoiced -> 0')

# 値域確認: すべて [0, 128]
f0_rand = np.random.uniform(50, 2000, 300)
result = tok.quantize_f0(f0_rand, 100)
assert result.min() >= 0 and result.max() <= 128, 'out of range'
print('PASS: value range [0, 128]')

# 出力長
result = tok.quantize_f0(np.zeros(267), 100)
assert len(result) == 100, f'expected 100, got {len(result)}'
print('PASS: output length')
"
```

### 4.3 E2Eテスト

```bash
uv run python -c "
import numpy as np
from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer
tok = PitchTokenizer()

# 880Hz → MIDI 81 (A5)
result = tok.quantize_f0(np.array([880.0] * 3), 1)
assert result[0].item() == 81, f'expected 81, got {result[0].item()}'
print('PASS: 880Hz -> 81')

# 220Hz → MIDI 57 (A3)
result = tok.quantize_f0(np.array([220.0] * 3), 1)
assert result[0].item() == 57, f'expected 57, got {result[0].item()}'
print('PASS: 220Hz -> 57')
"
```

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- **区間計算の丸め誤差**: `round(i * F0_FPS / ENCODEC_FPS)` による境界のズレが累積すると、出力長が `target_len` からズレる可能性がある。ループで逐次計算しているため、各フレームは独立して計算され累積ズレは発生しないが、`start == end` になるフレーム（区間幅ゼロ）が生じる可能性がある点に注意。
- **`math.log2(0)` の防止**: `voiced = frame[frame > 0]` で無声フレームを除外しているが、`len(voiced) == 0` のチェックを必ず実施すること。
- **numpy vs torch の型変換**: 入力は `np.ndarray`、出力は `torch.LongTensor`。変換忘れに注意。

### 5.2 レビュー項目

- [ ] `quantize_f0([440.0], 1)` → `tensor([69])` を確認
- [ ] `quantize_f0([0.0], 1)` → `tensor([0])` を確認
- [ ] 出力値が全て `[0, 128]` の範囲内であることをランダム入力で確認
- [ ] `target_len=0` を渡したときに空テンソルが返ること（クラッシュしない）

## 6. フェーズ振り返り: 一から作り直すとしたら

**ピッチ量子化方式（MIDI vs セント）**

MIDI 整数量子化は実装シンプルだが、低音域では周波数解像度が粗い（例: MIDI 48=130.8Hz, 49=138.6Hz、差 7.8Hz）。歌唱音声では低音域の表現も重要であり、セント単位（1セント = 1/100半音）の均等量子化のほうが物理的に均等な量子化となる。

一から設計するなら、`bins_per_semitone` パラメータを `__init__` で設定可能にし、`bins_per_semitone=1` (MIDI) がデフォルトでも `bins_per_semitone=2` (セント換算12セント/bin)等に切り替えられる設計にする。

**TDD の観点**

`quantize_f0` の受入基準（440Hz→69, 0Hz→0）は明確であり、先にテストを書いてから実装する TDD が自然にはまる。M1-06 のテストを先に書いてから本チケットを実装するほうが品質が高い。

**数値安定性**

`math.log2` は `float` 精度で計算する。`np.float64` として `f0` を受け取る場合は明示的に `float()` でキャストすること。

## 7. 後続タスクへの連絡事項

- **M1-06 (test_pitch_tokenizer)**: 受入基準として `quantize_f0([440.0], 1) → tensor([69])` と `quantize_f0([0.0], 1) → tensor([0])` の2ケースを必ずテストに含めること。また区間計算の境界値テスト（`f0` 長さ1, `target_len=1`）も追加推奨。
- **M1-14 (extract_pitch_tokens)**: pyworld DIO 出力は `frame_period=5ms` → 200Hz。`quantize_f0(f0, T_a)` で `T_a` は音響トークン長（75Hz 基準）を渡すこと。pyworld の F0 配列は `np.float64` であるため、型変換は不要（`quantize_f0` 内で処理済み）。
- **M1-03, M1-04**: `quantize_f0` とは独立して実装可能。本チケットの完了を待たずに着手できる。
