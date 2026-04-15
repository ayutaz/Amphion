# M1-18: preprocess.py 統合テスト

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-05
> **依存チケット**: M1-12, M1-13, M1-14, M1-15, M1-16, M1-17
> **ブロックするチケット**: M2（学習パイプライン前提）
> **状態**: TODO

---

## 1. 目的とゴール

`tests/test_preprocess.py` にて合成正弦波（440Hz, 2秒）を入力とし、M1-12〜17 の全関数を通して shape/dtype/値域/長さ整合を検証する統合テスト。

**ゴール**: `uv run pytest tests/test_preprocess.py -v -m "not slow"` 全件PASS（モックテスト）。`uv run pytest tests/test_preprocess.py -v` で slow テスト含む全件PASS。

## 2. 実装する内容の詳細

### テスト構成

```python
# tests/test_preprocess.py

def make_sine_wave(freq=440.0, duration=2.0, sr=24000):
    t = torch.linspace(0, duration, int(sr * duration))
    return torch.sin(2 * torch.pi * freq * t)

# --- モック版（高速） ---
def test_acoustic_tokens_shape_dtype(monkeypatch): ...
def test_upsample_50hz_to_75hz(): ...
def test_pitch_tokens_value_range(): ...
def test_phone_ids_basic(): ...
def test_save_load_roundtrip(tmp_path): ...
def test_save_preprocessed_length_mismatch(): ...

# --- 実モデル版（@pytest.mark.slow） ---
@pytest.mark.slow
def test_acoustic_tokens_real_model(): ...
@pytest.mark.slow
def test_semantic_tokens_real_model(): ...
@pytest.mark.slow
def test_pitch_tokens_sine_440hz_real(): ...
@pytest.mark.slow
def test_run_preprocess_small_dataset(tmp_path): ...
```

### 実行コマンド

```bash
# 高速テスト（モックのみ）
uv run pytest tests/test_preprocess.py -v -m "not slow"

# 全テスト（実モデル含む）
uv run pytest tests/test_preprocess.py -v
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当 |
|---|---|---|
| テスト実装 | 1 | テストファイル全体 |
| レビュー | 1 | 実モデルテストの動作確認 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲

- `tests/test_preprocess.py`（モック版6件 + slow版4件 = 10件）

### 4.2 ユニットテスト

| テスト | slow | 確認内容 |
|---|---|---|
| `test_acoustic_tokens_shape_dtype` | - | shape(T_a,12), dtype long, 値域 |
| `test_upsample_50hz_to_75hz` | - | interpolate 比率 |
| `test_pitch_tokens_value_range` | - | 値域[0,128] |
| `test_phone_ids_basic` | - | dtype, dim, 値域 |
| `test_save_load_roundtrip` | - | 全フィールド保存・ロード |
| `test_save_preprocessed_length_mismatch` | - | ValueError |

### 4.3 E2Eテスト

| テスト | slow | 確認内容 |
|---|---|---|
| `test_acoustic_tokens_real_model` | yes | 実 Codec で完走 |
| `test_semantic_tokens_real_model` | yes | 実 w2v-bert で shape 確認 |
| `test_pitch_tokens_sine_440hz_real` | yes | MIDI 69 変換 |
| `test_run_preprocess_small_dataset` | yes | 5件 end-to-end |

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- slow テストは実モデルのチェックポイントパスが必要。CI でのキャッシュ設定要
- `torchaudio.save` が CI 環境で利用可能か事前確認

### 5.2 レビュー項目

- [ ] モックテストと slow テストが明確に分離されているか
- [ ] `conftest.py` に共通フィクスチャ（`make_sine_wave` 等）が定義されているか
- [ ] テスト実行時間が合理的か（モック版 < 10秒、slow版 < 120秒）

## 6. フェーズ振り返り: 一から作り直すとしたら

- **テスト戦略の分離**: モデル依存とロジック純粋テストを明確に分離し CI の fast/slow レイヤーを設計
- **conftest.py 共有**: `make_sine_wave`, `mock_codec_model` 等の共通フィクスチャを定義
- **Property-based testing**: Hypothesis で音声長・話者数をランダム変化させてエッジケース自動探索

## 7. 後続タスクへの連絡事項

- このチケットが GREEN になって初めて学習パイプライン（M3）の実装を開始できる
- CI では `uv run pytest tests/test_preprocess.py -v -m "not slow"` を通常 PR チェック、`-m slow` は週次実行
- M1-12〜17 の実装者は本テスト全パスまでマージを控えること
