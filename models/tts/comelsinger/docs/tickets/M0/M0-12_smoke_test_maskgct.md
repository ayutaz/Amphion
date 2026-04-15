# M0-12: MaskGCT 既存コード動作確認（smoke test）

> **マイルストーン**: [M0: 環境構築](../../13_milestones.md#m0-環境構築)
> **対応RQ**: -（環境構築）
> **依存チケット**: M0-01, M0-02, M0-03, M0-04, M0-05, M0-06（注: M0-07〜M0-10のモデルダウンロードは不要。本チケットはインポート確認のみでモデル重みロードは含まない）
> **ブロックするチケット**: M0-13, M2-11
> **状態**: TODO

---

## 1. 目的とゴール

MaskGCT の既存モジュール（`maskgct_s2a.py`, `maskgct_t2s.py`, `llama_nar.py`, `maskgct_utils.py`）が正常にインポートでき、クラスのインスタンス化が実行できることを確認する。M0 フェーズの集大成として依存ライブラリ・コーデック・モデルコードの全体的な整合性を検証する。

## 2. 実装する内容の詳細

### 受入確認コマンド（最小要件）

```bash
uv run python -c "
from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A
from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S
print('OK')
"
```

### 拡張 smoke test

```python
# scripts/smoke_test_maskgct.py
import sys, torch

def smoke_test():
    errors = []
    try:
        from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A
        from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S
        print("[PASS] MaskGCT_S2A, MaskGCT_T2S imported")
    except ImportError as e:
        errors.append(f"[FAIL] Import error: {e}")

    try:
        from models.tts.maskgct.llama_nar import DiffLlama, DiffLlamaPrefix
        print("[PASS] DiffLlama, DiffLlamaPrefix imported")
    except ImportError as e:
        errors.append(f"[FAIL] llama_nar: {e}")

    try:
        from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
        print("[PASS] CodecEncoder, CodecDecoder imported")
    except ImportError as e:
        errors.append(f"[FAIL] codec: {e}")

    if torch.cuda.is_available():
        print(f"[INFO] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA not available (CPU-only mode)")

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)
    else:
        print("\nAll smoke tests passed.")

smoke_test()
```

### よくあるエラーと対処

| エラー | 原因 | 対処 |
|---|---|---|
| `No module named 'models.codec'` | M0-01 未完了 | `git sparse-checkout add models/codec` |
| `No module named 'transformers'` | M0-04 未完了 | `uv add transformers>=4.40.0` |
| `No module named 'json5'` | M0-06 未完了 | `uv add json5` |

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 環境構築担当 | 1 | smoke test 実行、エラー調査と修正 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: MaskGCT 全モジュールのインポート確認、主要依存ライブラリの確認

**含まないもの**: MaskGCT の推論実行（M4）、モデル重みのロード（M0-07〜08）

### 4.2 ユニットテスト

```bash
uv run python -c "
from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A
from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S
print('PASS')
"
```

### 4.3 E2Eテスト

```bash
uv run python scripts/smoke_test_maskgct.py
```

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- `maskgct_s2a.py` が Amphion 固有ライブラリをトップレベルでインポートしている場合、M0-06 完了が必須（高）
- `PYTHONPATH` 設定が `uv run` 環境で引き継がれない（中）
- `llama_nar.py` が transformers 内部 API に依存している場合のバージョン互換性（中）

### 5.2 レビュー項目

- [ ] 全インポートがエラーなく完了したか
- [ ] `models/codec/` のインポートが成功したか
- [ ] smoke test 出力ログを M0-13 担当者に共有したか

## 6. フェーズ振り返り: 一から作り直すとしたら

- smoke test はインポートだけでなく `load_state_dict` + ダミー forward まで含める
- `transformers`, `peft`, `torch` のバージョンを pin し smoke test でアサーション追加
- Dockerfile の `RUN` ステップに smoke test を組み込みビルド時に環境検証を自動化
- `pytest tests/test_smoke_maskgct.py` として CI の必須ステップに追加

## 7. 後続タスクへの連絡事項

- **M0-13**: PYTHONPATH 設定を引き継ぐ。comelsinger パッケージも同じ設定で動作するはず。
- **M2-11**: `MaskGCT_S2A` の `__init__` シグネチャを確認し `CoMelSinger_S2A` の継承設計に活用すること。
