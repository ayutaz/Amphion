# M0-13: comelsinger パッケージ初期化確認

> **マイルストーン**: [M0: 環境構築](../../13_milestones.md#m0-環境構築)
> **対応RQ**: -（環境構築）
> **依存チケット**: M0-12
> **ブロックするチケット**: M1 全体（M1-01〜M1-18）
> **状態**: TODO

---

## 1. 目的とゴール

`models/tts/comelsinger/__init__.py` が存在し、`from models.tts.comelsinger import *` がエラーなしで実行できることを確認する。M0 フェーズの最終チケットとして、M1 以降の実装チケットが安定してインポートできる状態を保証する。

## 2. 実装する内容の詳細

### `__init__.py` 確認・作成

```bash
ls -la models/tts/comelsinger/__init__.py
```

未存在の場合:

```python
# models/tts/comelsinger/__init__.py
"""
CoMelSinger: Discrete Token-Based Zero-Shot Singing Synthesis
With Structured Melody Control and Guidance.

Reference: arXiv:2509.19883v2
"""

__version__ = "0.1.0"
__all__ = []
```

### 受入確認

```bash
uv run python -c "from models.tts.comelsinger import *"
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 環境構築担当 | 1 | `__init__.py` 確認・作成、インポート確認 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: `__init__.py` の存在確認・作成、インポート動作確認

**含まないもの**: 各サブモジュールの実装（M1 の責務）

### 4.2 ユニットテスト

```bash
uv run python -c "
import os
assert os.path.exists('models/tts/comelsinger/__init__.py')
from models.tts.comelsinger import *
import models.tts.comelsinger as pkg
assert pkg.__name__ == 'models.tts.comelsinger'
print('OK')
"
```

### 4.3 E2Eテスト

M1-01 の `PitchTokenizer` が `from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer` でインポートできることが M1 の E2E 確認となる。

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- `__init__.py` に壊れたインポートが含まれている可能性（中）
- `from models.tts.comelsinger import *` が `docs/` 以下を走査してエラーを起こす可能性（低）→ `__all__ = []` で防止

### 5.2 レビュー項目

- [ ] `models/tts/comelsinger/__init__.py` が存在するか
- [ ] `from models.tts.comelsinger import *` が exit code 0 か
- [ ] `__all__` が定義されているか

## 6. フェーズ振り返り: 一から作り直すとしたら

- `__init__.py` に記載するモデルバージョン定数（`MODEL_REVISION = "abc1234"`）を一元管理する設計
- `config/comelsinger_config.yaml` でモデルバージョン・パスを一元管理し、`__init__.py` はそのロードのみ行う構成
- M1〜M4 の各実装完了後に自動でインポートテストが実行されるよう CI に追加

## 7. 後続タスクへの連絡事項

- **M1-01**: `pitch_tokenizer.py` 新規作成時は `__init__.py` に `from .pitch_tokenizer import PitchTokenizer` を追加すること。
- **M1 全体**: 新規ファイル追加のたびに `__init__.py` のエクスポートリスト更新を検討すること。
- **M0 フェーズ完了**: 本チケット完了をもって M0 全体が完了。`INDEX.md` の M0 進捗を更新すること。
