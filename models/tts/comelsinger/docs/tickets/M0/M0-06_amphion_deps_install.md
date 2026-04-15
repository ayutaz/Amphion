# M0-06: Amphion 固有依存インストール

> **マイルストーン**: [M0: 環境構築](../../13_milestones.md#m0-環境構築)
> **対応RQ**: -（環境構築）
> **依存チケット**: M0-03
> **ブロックするチケット**: M0-12
> **状態**: TODO

---

## 1. 目的とゴール

Amphion の MaskGCT コードは `json5`・`ruamel.yaml`・`tqdm`・`huggingface_hub` を使用する。特に `json5` は JSON5 形式設定ファイル読み込みに必須。

## 2. 実装する内容の詳細

```bash
uv add json5 "ruamel.yaml" tqdm huggingface_hub
```

### HF_TOKEN 設定

```bash
echo "HF_TOKEN=your_token_here" >> .env
# .gitignore に .env を追加すること
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実行エージェント | 1 | uv add 実行・動作確認 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: 4ライブラリのインストール、HF_TOKEN 設定案内

**含まないもの**: モデルダウンロード（→ M0-07〜M0-10）、Amphion 設定ファイル作成（→ M3-01）

### 4.2 ユニットテスト

```bash
uv run python -c "
import json5
config = json5.loads('{\"a\": 1, // comment\n\"b\": 2,}')
assert config['a'] == 1
print('PASS')
"
```

### 4.3 E2Eテスト

```bash
uv run huggingface-cli --help | head -5
```

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- **json5 と標準 json の混在**: Amphion のコードが両方使う場合があり設定形式に注意。
- **HF_TOKEN の管理**: `.env` にトークンを含める場合は `.gitignore` に追加必須。

### 5.2 レビュー項目

- [ ] `import json5` 成功、コメント付き JSON5 パース可能
- [ ] `uv run huggingface-cli --help` 動作
- [ ] `.env` が `.gitignore` に追加されているか

## 6. フェーズ振り返り: 一から作り直すとしたら

**設定ファイル形式**: Amphion は JSON5 を採用（コメント可）。一から設計するなら `pyproject.toml` の `[tool.comelsinger]` に設定を集約する設計が管理しやすい。

**CI/CD での再現性**: `uv sync --frozen` で `uv.lock` を変更せず完全再現。

## 7. 後続タスクへの連絡事項

- **M0-07〜M0-10**: `huggingface_hub` と `HF_TOKEN` が必要。
- **M0-12**: `json5` が Amphion 設定読込に使われる可能性あり。
- **M3-01**: JSON5 形式の設定ファイルを作成する。
