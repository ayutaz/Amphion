# M0-01: sparse-checkout に models/codec 追加

> **マイルストーン**: [M0: 環境構築](../../13_milestones.md#m0-環境構築)
> **対応RQ**: -（環境構築）
> **依存チケット**: -
> **ブロックするチケット**: M0-10, M0-12
> **状態**: TODO

---

## 1. 目的とゴール

MaskGCT の `maskgct_utils.py` は `CodecEncoder` / `CodecDecoder` を `from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder` でインポートする。現在の sparse-checkout 設定に `models/codec` が含まれていないため、このインポートは `ModuleNotFoundError` で失敗する。

このチケットを完了すると `models/codec/` がワーキングツリーに展開され、後続のすべての Python インポートと smoke test が通る状態になる。

## 2. 実装する内容の詳細

### 手順

```bash
# 1. 現在の sparse-checkout リストを確認
git sparse-checkout list

# 2. models/codec を追加
git sparse-checkout add models/codec

# 3. ファイルが展開されたか確認
ls models/codec/amphion_codec/codec.py

# 4. インポート確認
uv run python -c "from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder; print('OK')"
```

### 注意事項

- `git sparse-checkout add` は既存のパターンを保持したまま追加する。`git sparse-checkout set` を誤って実行すると既存のチェックアウト済みパスが消えるため、必ず `add` を使用すること。
- sparse-checkout の設定ファイルは `.git/info/sparse-checkout` に保存される。変更後は `git sparse-checkout list` で内容を確認すること。

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実行エージェント | 1 | git コマンド実行・確認コマンド実行 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: `git sparse-checkout add models/codec` の実行、展開結果の確認

**含まないもの**: `models/codec` のコード自体への変更、codec モデル重みのダウンロード（→ M0-10）

### 4.2 ユニットテスト

```bash
uv run python -c "
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
print('CodecEncoder:', CodecEncoder)
print('CodecDecoder:', CodecDecoder)
print('PASS')
"
```

### 4.3 E2Eテスト

```bash
git sparse-checkout list | grep "models/codec"
ls models/codec/amphion_codec/
```

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- **部分的なチェックアウト失敗**: ネットワーク障害やリポジトリ設定で一部ファイルのみ取得される場合がある。`ls` で必要ファイルの存在を必ず確認すること。
- **cone mode vs no-cone mode**: sparse-checkout の mode によりパターン指定方法が異なる。`git sparse-checkout list` で確認。

### 5.2 レビュー項目

- [ ] `git sparse-checkout list` に `models/codec` が含まれているか
- [ ] `models/codec/amphion_codec/codec.py` が存在するか
- [ ] インポートテストが例外なく通るか

## 6. フェーズ振り返り: 一から作り直すとしたら

**sparse-checkout vs full clone のトレードオフ**

sparse-checkout は大規模モノレポ（Amphion のように多数のモデルを含むリポジトリ）でディスク使用量とクローン時間を削減するために有効だが、後から必要なパスを発見するたびに `git sparse-checkout add` が必要になるコストがある。

一から設計し直すとしたら、開発初期は **full clone** にして必要なパスを洗い出した上で、本番 CI 向けに sparse-checkout パターンを確定する。モノレポの総サイズが数十 GB を超えない限り、開発者体験を優先して full clone を推奨する。

**PYTHONPATH 管理**: `models/codec` のような相対インポートが機能するには、リポジトリルートが `PYTHONPATH` に含まれている必要がある。`pyproject.toml` で PYTHONPATH を設定するか `.env` に `PYTHONPATH=.` を追加する。

## 7. 後続タスクへの連絡事項

- **M0-10**: `models/codec` のコードパスが前提。このチケット未完了では M0-10 は開始できない。
- **M0-12**: `from models.codec.amphion_codec.codec import ...` が内部的に呼ばれる。このチケット未完了では smoke test は失敗する。
- `git sparse-checkout list` の出力内容を M0-12 担当者に共有すること。
