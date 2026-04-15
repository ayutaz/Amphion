# M0-02: uv init でプロジェクト初期化

> **マイルストーン**: [M0: 環境構築](../../13_milestones.md#m0-環境構築)
> **対応RQ**: -（環境構築）
> **依存チケット**: -
> **ブロックするチケット**: M0-03, M0-04, M0-05, M0-06
> **状態**: TODO

---

## 1. 目的とゴール

リポジトリルートに `pyproject.toml` が存在しない状態では `uv add` が機能せず、以降の全依存インストールチケット（M0-03〜M0-06）が実行できない。

このチケットを完了すると `pyproject.toml` と `uv.lock` がリポジトリルートに生成され、`uv run python --version` が正常に動作する状態になる。

## 2. 実装する内容の詳細

### 手順

```bash
# 1. リポジトリルートにいることを確認
pwd

# 2. 既存の pyproject.toml の有無を確認
ls pyproject.toml 2>/dev/null && echo "already exists" || echo "not found"

# 3. uv init 実行
uv init --name comelsinger --no-package

# 4. PYTHONPATH 設定（.env ファイル作成）
echo "PYTHONPATH=." > .env

# 5. 動作確認
uv run python --version
```

### pyproject.toml の初期設定

```toml
[project]
name = "comelsinger"
version = "0.1.0"
description = "CoMelSinger reproduction based on Amphion/MaskGCT"
requires-python = ">=3.10"
```

### 注意事項

- `--no-package` フラグにより `src/` レイアウトへの変換を防ぐ。Amphion は `models/` を直接インポートするフラットな構造のため必須。
- `.env` に `PYTHONPATH=.` を設定することで `uv run python -c "from models.xxx import ..."` が機能する。

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実行エージェント | 1 | uv init 実行・設定ファイル編集・動作確認 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: `uv init` 実行、`pyproject.toml` 生成、PYTHONPATH 設定（`.env`）

**含まないもの**: ライブラリの追加（→ M0-03〜M0-06）、GPU ドライバ・CUDA のインストール

### 4.2 ユニットテスト

```bash
uv run python --version
# 期待: Python 3.10.x 以上

test -f pyproject.toml && echo "PASS" || echo "FAIL"

uv run python -c "import sys; print('.' in sys.path or '' in sys.path)"
# 期待: True
```

### 4.3 E2Eテスト

```bash
uv run python -c "print('uv run is working')"
cat .env
# 期待: PYTHONPATH=. が含まれる
```

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- **既存の pyproject.toml との競合**: Amphion リポジトリに既存の `pyproject.toml` や `setup.py` がある場合、`uv init` が上書きするリスクがある。事前に存在確認を行うこと。
- **Python バージョンの不一致**: システムの Python が 3.10 未満の場合、`uv run` が失敗する。`uv python install 3.10` で対応可能。

### 5.2 レビュー項目

- [ ] `pyproject.toml` が生成されているか
- [ ] `requires-python = ">=3.10"` が設定されているか
- [ ] `uv run python --version` が 3.10 以上を返すか
- [ ] PYTHONPATH に `.` が含まれているか

## 6. フェーズ振り返り: 一から作り直すとしたら

**uv vs poetry vs pip の選択**

| ツール | メリット | デメリット |
|---|---|---|
| uv | 高速、lock ファイルで完全再現性、Python バージョン管理も統合 | 比較的新しく事例が少ない |
| poetry | 成熟、エコシステムが豊富 | 低速、conda との相性が悪い |
| pip + requirements.txt | シンプル、広く普及 | 再現性が弱い |

研究プロジェクトでは再現性が最重要であり、`uv.lock` による完全なピン留めと高速インストールを両立する `uv` が最適と判断した。

**再現性確保の設計**: `pyproject.toml` + `uv.lock` をリポジトリにコミットする。CI/CD では `uv sync --frozen` で完全再現できる。

**`--no-package` の重要性**: Amphion は `models/` を src レイアウトなしで直接インポートするため必須。一から設計するなら `src/comelsinger/` レイアウトに移行し、Amphion の既存コードも PYTHONPATH 設定で吸収する設計が管理しやすい。

## 7. 後続タスクへの連絡事項

- **M0-03〜M0-06**: `pyproject.toml` が存在しない状態で `uv add` を実行するとエラーになる。
- `.env` に `PYTHONPATH=.` を設定したことで、後続の受入確認コマンドが正しく動作する。
- Python バージョンの確認結果を M0-03 担当者に共有すること（CUDA ビルドの選択に影響）。
