# M1-01: PitchTokenizer クラス骨格

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-01
> **依存チケット**: -
> **ブロックするチケット**: M1-02, M1-03, M1-04, M1-05, M1-06
> **状態**: TODO

---

## 1. 目的とゴール

`pitch_tokenizer.py` に `PitchTokenizer` クラスの骨格を作成する。定数 `VOCAB_SIZE=129`・`ENCODEC_FPS=75.0`・`F0_FPS=200.0` を定義し、`__init__` のみ実装する。他のメソッドはすべて `NotImplementedError` を送出するスタブとして定義する。

このチケットを完了すると、後続チケット（M1-02〜M1-05）が各メソッドを独立して実装できる土台が整う。

## 2. 実装する内容の詳細

### 2.1 ファイル配置

```
models/tts/comelsinger/pitch_tokenizer.py  ← 新規作成
```

### 2.2 クラス骨格

```python
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch

class PitchTokenizer:
    """F0連続値またはMIDI楽譜からフレームアラインド離散ピッチトークンを生成"""

    VOCAB_SIZE: int = 129       # 0: unvoiced, 1-128: MIDI note 1-128
    ENCODEC_FPS: float = 75.0   # Amphion Codec のフレームレート (24kHz/320)
    F0_FPS: float = 200.0       # pyworld DIO frame_period=5ms

    def __init__(
        self,
        vocab_size: int = 129,
        encodec_fps: float = 75.0,
        f0_fps: float = 200.0,
    ) -> None:
        self.vocab_size = vocab_size
        self.encodec_fps = encodec_fps
        self.f0_fps = f0_fps

    def quantize_f0(self, f0: np.ndarray, target_len: int) -> torch.LongTensor:
        """F0 (T_f0,) Hz → ピッチトークン (L,)"""
        raise NotImplementedError

    def tokenize_score(
        self,
        note_midi: List[int],
        duration_symbol: List[int],
        target_len: int,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """楽譜 → (ピッチトークン (L,), フレーム数列 (S,))"""
        raise NotImplementedError

    def compute_soft_label_matrix(
        self, m_p: torch.LongTensor
    ) -> torch.FloatTensor:
        """(L,) → (L,L) ソフトラベル行列"""
        raise NotImplementedError

    def decode_token_to_freq(self, token: int) -> float:
        """ピッチトークン → Hz 逆変換"""
        raise NotImplementedError
```

### 2.3 注意事項

- クラス定数（大文字）とインスタンス属性（小文字）の両方を定義する。クラス定数は型ヒント付きで定義し、`mypy` の型チェックで引っかからないようにする。
- `NotImplementedError` のメッセージには「どのチケットで実装するか」を記載する（例: `"Implemented in M1-02"`）。

**重要: Python は全て `uv run python` / `uv run pytest` で実行。pip 禁止。**

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `pitch_tokenizer.py` 骨格の作成 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: `pitch_tokenizer.py` の新規作成、`PitchTokenizer.__init__` の実装、4つのスタブメソッドの定義

**含まないもの**: 各メソッドの実際の実装（→ M1-02〜M1-05）、テストファイル（→ M1-06）

### 4.2 ユニットテスト

```bash
uv run python -c "
from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer
tok = PitchTokenizer()
assert tok.vocab_size == 129
assert tok.encodec_fps == 75.0
assert tok.f0_fps == 200.0
print('PASS: __init__ OK')

# スタブが NotImplementedError を送出するか確認
import numpy as np, torch
try:
    tok.quantize_f0(np.array([440.0]), 1)
    assert False, 'should raise'
except NotImplementedError:
    print('PASS: quantize_f0 stub OK')
"
```

### 4.3 E2Eテスト

```bash
uv run python -c "
from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer
tok = PitchTokenizer()
assert PitchTokenizer.VOCAB_SIZE == 129
assert PitchTokenizer.ENCODEC_FPS == 75.0
assert PitchTokenizer.F0_FPS == 200.0
print('PASS: class constants OK')
"
```

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- **`models/tts/comelsinger/__init__.py` の存在確認**: `PitchTokenizer` をパッケージとしてインポートするには `__init__.py` が必要。M0-13 で確認済みのはずだが、ファイル作成前に確認すること。
- **メソッドシグネチャの凍結**: 骨格チケットでシグネチャを確定するため、M1-02〜05 の実装担当者はシグネチャを変更しないこと。変更が必要な場合はこのチケットを修正した上で関係者に周知する。

### 5.2 レビュー項目

- [ ] `PitchTokenizer()` がエラーなくインスタンス化できるか
- [ ] `VOCAB_SIZE=129`, `ENCODEC_FPS=75.0`, `F0_FPS=200.0` の値が正しいか
- [ ] 4つのスタブメソッドが `NotImplementedError` を送出するか
- [ ] 型ヒントが全メソッドに付与されているか

## 6. フェーズ振り返り: 一から作り直すとしたら

**ピッチ量子化方式（MIDI vs セント）**

MIDI整数量子化は実装がシンプルだが、MIDI番号間隔は指数的（半音=約6%周波数差）であり、低音域ほど粗くなる。セント単位での均等量子化（100セント=1半音、12セント/bin等）のほうが全音域で均等な精度になる。

一から設計するなら、`MIDI_BINS` と `CENT_BINS` の両モードを骨格段階でインターフェースとして定義し、実験で優れたほうを選択できる設計にする。

**TDD（テスト駆動開発）の観点**

骨格チケットより先にテストチケット（M1-06）の受入基準を確定し、それを満たすシグネチャを設計する TDD アプローチが理想的。今回は M1-01 → M1-06 の順序だが、実際には M1-06 の受入基準を先に書いてから M1-01 のシグネチャを固める。

**型安全性**

`np.ndarray` と `torch.LongTensor` の混在はバグの温床になりやすい。骨格段階から `torch.Tensor` のみに統一するか、入力型を明示的に変換するラッパーを追加する設計を検討する。

## 7. 後続タスクへの連絡事項

- **M1-02**: `quantize_f0` のシグネチャ `(f0: np.ndarray, target_len: int) -> torch.LongTensor` を厳守すること。
- **M1-03**: `tokenize_score` のシグネチャ `(note_midi, duration_symbol, target_len) -> (tokens, frame_counts)` を厳守すること。
- **M1-04**: `compute_soft_label_matrix` の入力は `torch.LongTensor` で `(L,)` 形状。
- **M1-05**: `decode_token_to_freq` は `int` を受け取り `float` を返す。
- **M1-06**: テスト作成時は `from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer` でインポートすること。`PYTHONPATH=.` が設定されているか事前確認が必要。
