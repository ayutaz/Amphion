# M1-06: PitchTokenizer 単体テスト

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-01
> **依存チケット**: M1-02, M1-03, M1-04, M1-05
> **ブロックするチケット**: M2-01, M2-02
> **状態**: TODO

---

## 1. 目的とゴール

`tests/test_pitch_tokenizer.py` を作成し、M1-02〜M1-05 の全受入基準をカバーする完全なテストスイートを構築する。境界値テスト・ラウンドトリップテスト・E2E テストを含む。

**ゴール**: `uv run pytest tests/test_pitch_tokenizer.py -v` が全テストパス

## 2. 実装する内容の詳細

### テストクラス構成

- `TestPitchTokenizerInit`: 定数確認（VOCAB_SIZE=129, ENCODEC_FPS=75.0, F0_FPS=200.0）
- `TestQuantizeF0`: 440Hz→69, 0Hz→0, 混合有声/無声, clamp境界, 出力長一致
- `TestTokenizeScore`: 均等/不均等デュレーション, sum保証, min>=1保証, 休符トークン
- `TestComputeSoftLabelMatrix`: 同一ピッチ→1.0, 無声同士→0, 対称性, dtype=float32
- `TestDecodeTokenToFreq`: A4→440Hz, 無声→0.0, オクターブ関係, 境界値, ValueError
- `TestRoundTrip`: 全有声トークン(1-128)のdecode→quantizeラウンドトリップ
- `TestE2E`: tokenize_score→compute_soft_label_matrixのパイプライン

### テスト実行

```bash
uv run pytest tests/test_pitch_tokenizer.py -v
uv run pytest tests/test_pitch_tokenizer.py --cov=models.tts.comelsinger.pitch_tokenizer
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当 |
|---|---|---|
| テスト実装 | 1 | テストファイル全体の作成・実行確認 |
| テストレビュー | 1 | 受入基準との対応確認・境界値補完 |
| CI設定 | 1 | pyproject.toml の pytest 設定追加 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲

- `tests/test_pitch_tokenizer.py`（50+ テスト関数）
- `pyproject.toml` の pytest 設定更新

### 4.2 ユニットテスト

50+ テスト関数で M1-02〜05 の全受入基準をカバー。`torch.manual_seed(42)` で再現性確保。

### 4.3 E2Eテスト

- `tokenize_score → compute_soft_label_matrix` のパイプラインテスト
- 全トークン(0-128)のデコードバッチテスト

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- ラウンドトリップテストで `round()` の銀行丸めにより一部 MIDI で失敗する可能性
- E2E テストで `pyworld` が必要な場合、CI 環境での依存追加が必要

### 5.2 レビュー項目

- [ ] 各テスト関数が受入基準と1:1で対応しているか
- [ ] 境界値（token=0, token=128, target_len=1）が網羅されているか
- [ ] カバレッジ 100% が達成されているか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **TDD 徹底**: 骨格作成と同時にテストを `xfail` マークで先行作成すべきだった
- **Property-based testing**: `hypothesis` で「任意の有声 MIDI トークンのラウンドトリップ成立」を形式検証
- **conftest.py**: 共通 fixture を分離してテストファイル肥大化を防ぐ

## 7. 後続タスクへの連絡事項

- **M2-01**: `tests/` のテスト構造を参考に `test_svt_module.py` を作成
- **CI**: `pyproject.toml` に `[tool.pytest.ini_options] testpaths = ["tests"]` を追加
- M2 以降で pitch_tokenizer を変更する場合、本テストが全パスすることを必須条件とする
