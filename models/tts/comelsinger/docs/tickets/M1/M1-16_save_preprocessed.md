# M1-16: .ptファイル保存関数 (save_preprocessed)

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-05
> **依存チケット**: M1-12, M1-13, M1-14, M1-15
> **ブロックするチケット**: M1-17, M1-18
> **状態**: TODO

---

## 1. 目的とゴール

M1-12〜15 で抽出した全フィールドを辞書にまとめて `.pt` ファイルに保存する。保存前に長さ整合チェックを行い不整合の場合は `ValueError` を投げる。

**ゴール**: `acoustic_tokens` は `(12, T_a)` に転置保存。保存→ロードで全テンソルが一致。

## 2. 実装する内容の詳細

```python
def save_preprocessed(
    output_path: str,
    acoustic_tokens: torch.Tensor,    # (T_a, 12) → (12, T_a) に転置
    semantic_tokens: torch.Tensor,    # (T_a,)
    pitch_tokens: torch.Tensor,       # (T_a,)
    phone_ids: torch.Tensor,          # (T_ph,)
    note_durations: torch.Tensor,     # (S,)
    note_pitches: torch.Tensor,       # (S,)
    attention_mask: torch.Tensor,     # (T_a,)
    speaker_id: int,
) -> None:
    T_a = acoustic_tokens.shape[0]
    if semantic_tokens.shape[0] != T_a:
        raise ValueError(f"semantic_tokens length {semantic_tokens.shape[0]} != T_a {T_a}")
    if pitch_tokens.shape[0] != T_a:
        raise ValueError(f"pitch_tokens length {pitch_tokens.shape[0]} != T_a {T_a}")
    if attention_mask.shape[0] != T_a:
        raise ValueError(f"attention_mask length {attention_mask.shape[0]} != T_a {T_a}")

    data = {
        "acoustic_tokens": acoustic_tokens.permute(1, 0).contiguous(),  # (12, T_a)
        "semantic_tokens": semantic_tokens,
        "pitch_tokens": pitch_tokens,
        "phone_ids": phone_ids,
        "note_durations": note_durations,
        "note_pitches": note_pitches,
        "attention_mask": attention_mask,
        "speaker_id": torch.tensor(speaker_id, dtype=torch.long),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当 |
|---|---|---|
| 実装 | 1 | 保存関数・ロード関数 |
| レビュー | 1 | フィールド仕様・転置方向の確認 |

## 4. 提供範囲とテスト項目

### 4.2 ユニットテスト

- 保存→ロードで全フィールド一致（ラウンドトリップ）
- `acoustic_tokens` が `(12, T_a)` shape で保存されていること
- 長さ不整合で `ValueError` が発生すること

### 4.3 E2Eテスト

- M1-18 統合テストでラウンドトリップを全フィールドについて確認

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- `acoustic_tokens` の転置保存 `(T_a, 12) → (12, T_a)` は DataLoader の入力形式に合わせたもの。M2-09 の collate_fn と整合を確認
- `note_durations` の総和が `T_a` と一致するかのチェックも追加すべきか

### 5.2 レビュー項目

- [ ] `permute(1, 0)` の転置方向が正しいか
- [ ] `Path.mkdir(parents=True)` でディレクトリが自動作成されるか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **HDF5 形式**: 部分読み込み可能でメモリ効率が向上
- **WebDataset**: 大規模データセットで分散学習時の I/O ボトルネック解消
- **スキーマ検証**: Pydantic/dataclass でデータスキーマを定義し型チェック自動化
- **Arrow 形式**: HuggingFace datasets と統合可能

## 7. 後続タスクへの連絡事項

- **M1-17**: 出力ディレクトリ構造（`{split}/{speaker_id}/{utt_id}.pt`）を統一すること
- **M2-09**: DataLoader は `acoustic_tokens` を `(12, T_max)` shape で受け取る前提
- `load_preprocessed` 関数も `preprocess.py` に含めること
