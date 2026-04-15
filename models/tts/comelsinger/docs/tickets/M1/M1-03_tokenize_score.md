# M1-03: tokenize_score 実装

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-01
> **依存チケット**: M1-01
> **ブロックするチケット**: M1-06
> **状態**: TODO

---

## 1. 目的とゴール

`PitchTokenizer.tokenize_score()` を実装する。MusicXML / MIDI 楽譜から得られた音符列（MIDI 番号列 + デュレーション列）を受け取り、各フレームに対応する離散ピッチトークン列 `(L,)` と各音符のフレーム数列 `(S,)` を返す。

`sum(frame_counts) == target_len` および `all(frame_counts >= 1)` の両方を全ケースで保証することが最大の要件。

## 2. 実装する内容の詳細

### 2.1 アルゴリズム詳細

デュレーション正規化に **Largest Remainder Method（最大余剰法）** を使用する。

```
入力:
  note_midi: [m_1, m_2, ..., m_S]     整数MIDI番号（休符=0）
  duration_symbol: [d_1, d_2, ..., d_S]  シンボリックデュレーション（相対単位）
  target_len: L                          目標出力フレーム数

処理:
  D = sum(duration_symbol)               総デュレーション
  a_d_raw[i] = d_i / D * L              連続値フレーム割当

  Largest Remainder Method:
    a_d[i] = floor(a_d_raw[i])          整数部
    残差 remainder[i] = a_d_raw[i] - a_d[i]
    deficit = L - sum(a_d)              不足フレーム数
    deficit 個の最大余剰を +1           割り当て調整
    各 a_d[i] = max(a_d[i], 1)          最低1フレーム保証

  出力ピッチトークン列:
    tokens = repeat_interleave(note_midi, a_d)
    ← tokens.shape == (L,)

  出力フレーム数列:
    frame_counts = tensor(a_d)
    ← frame_counts.shape == (S,)
```

### 2.2 実装コード

```python
def tokenize_score(
    self,
    note_midi: List[int],
    duration_symbol: List[int],
    target_len: int,
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    楽譜 → (ピッチトークン (L,), フレーム数列 (S,))

    Args:
        note_midi: MIDI番号列 (S,)。休符は 0。
        duration_symbol: 相対デュレーション列 (S,)。整数、正値のみ。
        target_len: 目標出力長 L (= 音響トークンフレーム数 T_a)

    Returns:
        tokens: shape (L,), dtype torch.long
        frame_counts: shape (S,), dtype torch.long
                      sum(frame_counts)==target_len, all(frame_counts>=1)
    """
    S = len(note_midi)
    D = sum(duration_symbol)
    assert D > 0, "duration_symbol の合計が 0 です"

    # 各音符のフレーム数（連続値）
    a_d_raw = [d / D * target_len for d in duration_symbol]

    # Largest Remainder Method
    a_d = [int(x) for x in a_d_raw]
    remainders = [(a_d_raw[i] - a_d[i], i) for i in range(S)]
    deficit = target_len - sum(a_d)
    # 余剰が大きい順に +1 を割り当て
    remainders.sort(key=lambda x: -x[0])
    for _, idx in remainders[:deficit]:
        a_d[idx] += 1

    # 各フレームを最低 1 に保証（調整が必要な場合は末尾から削る）
    for i in range(S):
        if a_d[i] < 1:
            a_d[i] = 1
    # 合計が target_len からズレた場合の補正（最低保証により生じる場合）
    diff = sum(a_d) - target_len
    if diff > 0:
        # 余剰をフレームが多い順に削る
        order = sorted(range(S), key=lambda i: -a_d[i])
        for idx in order:
            if diff <= 0:
                break
            if a_d[idx] > 1:
                subtract = min(a_d[idx] - 1, diff)
                a_d[idx] -= subtract
                diff -= subtract

    # ピッチトークン列の構築
    token_list = []
    for midi, count in zip(note_midi, a_d):
        tok = max(0, min(128, midi))  # 範囲クランプ（休符=0 を保持）
        token_list.extend([tok] * count)

    tokens = torch.tensor(token_list, dtype=torch.long)
    frame_counts = torch.tensor(a_d, dtype=torch.long)

    assert tokens.shape[0] == target_len, \
        f"tokens length {tokens.shape[0]} != target_len {target_len}"
    assert frame_counts.sum().item() == target_len, \
        f"frame_counts sum {frame_counts.sum().item()} != target_len {target_len}"
    assert (frame_counts >= 1).all(), "frame_counts に 0 以下の値があります"

    return tokens, frame_counts
```

### 2.3 エッジケース処理

| 入力条件 | 処理 |
|---|---|
| `S=1` (音符1つ) | `a_d[0] = target_len`（全フレームを1音符に割り当て）|
| `duration_symbol` が全て等値 | 均等分割 + LRM による端数補正 |
| `target_len < S` | 全音符に最低1フレーム割り当てた後、余剰を削って合計を合わせる |
| 休符 (`note_midi[i]=0`) | token 0 として出力（無声と同一トークン）|

**重要: Python は全て `uv run python` / `uv run pytest` で実行。pip 禁止。**

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `tokenize_score` メソッドの実装 |
| レビューエージェント | 1 | LRM の正確性・境界値テスト設計 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: `tokenize_score` メソッドの完全実装（M1-01 の `NotImplementedError` を置き換え）

**含まないもの**: テストファイル作成（→ M1-06）、楽譜パーサー（楽譜はすでに MIDI 番号列・デュレーション列に変換済みを想定）

### 4.2 ユニットテスト

```bash
uv run python -c "
import random
from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer
tok = PitchTokenizer()

# 基本ケース
tokens, fc = tok.tokenize_score([69, 72, 76], [1, 1, 1], 12)
assert tokens.shape[0] == 12, f'expected 12, got {tokens.shape[0]}'
assert fc.sum().item() == 12
assert (fc >= 1).all()
print('PASS: basic case')

# ランダム100ケース
for _ in range(100):
    S = random.randint(1, 20)
    midi = [random.randint(48, 84) for _ in range(S)]
    dur  = [random.randint(1, 8) for _ in range(S)]
    L    = random.randint(S, 300)
    tokens, fc = tok.tokenize_score(midi, dur, L)
    assert tokens.shape[0] == L, f'length mismatch: {tokens.shape[0]} vs {L}'
    assert fc.sum().item() == L, f'sum mismatch: {fc.sum().item()} vs {L}'
    assert (fc >= 1).all(), 'frame_count < 1 found'
print('PASS: 100 random cases')
"
```

### 4.3 E2Eテスト

```bash
uv run python -c "
from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer
tok = PitchTokenizer()

# target_len == S (極小ケース)
tokens, fc = tok.tokenize_score([60, 62, 64], [1, 1, 1], 3)
assert tokens.shape[0] == 3
assert (fc >= 1).all()
print('PASS: target_len==S')

# 不均等デュレーション
tokens, fc = tok.tokenize_score([60, 64], [3, 1], 8)
assert tokens.shape[0] == 8
assert fc[0].item() >= fc[1].item()  # 長い音符は多いフレーム
print('PASS: unequal duration')
"
```

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- **`target_len < S` の場合**: 音符数よりフレーム数が少ない場合、全音符に最低1フレームを割り当てると `sum(a_d) > target_len` になる。この場合は最大フレーム数の音符から順にフレームを削る補正が必要。実装コードに含まれているが、テストで確認すること。
- **浮動小数点の精度**: `d / D * target_len` の計算で Python `float` を使用。大きな `D` や `target_len` では精度が落ちる可能性があるが、実用範囲（S≤100, L≤2000程度）では問題ない。
- **MIDI 範囲外の入力**: `note_midi` に 0〜128 範囲外の値が来た場合、`clamp` で対処しているが、上流の楽譜パーサーで検証することを推奨する。

### 5.2 レビュー項目

- [ ] 100ケースのランダムテストで `sum(frame_counts)==target_len` が全て成立するか
- [ ] `(frame_counts >= 1).all()` が全ケースで成立するか
- [ ] `target_len < S` のエッジケースでクラッシュしないか
- [ ] `S=1` のケースで正しく動作するか

## 6. フェーズ振り返り: 一から作り直すとしたら

**Largest Remainder Method の選択理由**

デュレーション正規化のシンプルな実装は `round(d/D*L)` だが、これは `sum(a_d) != target_len` になるケースが頻繁に生じる。LRM は選挙制度での議席配分問題と同一の構造であり、整数割り当て問題の「最良の単純解」として知られている。

一から設計するなら、まず `round` ベースの単純実装を書いてテストで `sum != target_len` のケースを観察し、そこから LRM に移行する TDD サイクルを踏む。

**型安全性**

`duration_symbol: List[int]` とドキュメントしているが、MIDI ライブラリによっては `float` が来ることがある。入力検証（`assert all(d > 0 for d in duration_symbol)`）を `__init__` ではなく各メソッドの先頭で行う設計にする。

**楽譜形式の抽象化**

`note_midi + duration_symbol` の2リスト形式は最もシンプルだが、実際の楽譜には装飾音符（グレースノート）・タイ・スラーなど複雑な構造が含まれる。`Score` データクラスを設計して `tokenize_score(score: Score, target_len)` とするほうが拡張性が高い。

## 7. 後続タスクへの連絡事項

- **M1-06 (test_pitch_tokenizer)**: `tokenize_score` のテストでは必ずランダム100ケース以上のテストを含め、`sum(frame_counts)==target_len` と `all(frame_counts>=1)` の両方を検証すること。
- **M1-04 (compute_soft_label_matrix)**: `tokenize_score` の出力 `tokens (L,)` が `compute_soft_label_matrix` の入力 `m_p (L,)` として直接渡せるよう、同じ dtype（`torch.long`）を使用する。
- **M2-7 (CoMelSingerDataset)**: データセットの `note_durations` フィールドに `frame_counts (S,)` をそのまま格納することを想定している。
