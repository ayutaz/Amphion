# M0-09: 事前学習済みモデル: w2v-bert-2.0

> **マイルストーン**: [M0: 環境構築](../../13_milestones.md#m0-環境構築)
> **対応RQ**: -（環境構築）
> **依存チケット**: なし
> **ブロックするチケット**: M1-13
> **状態**: TODO

---

## 1. 目的とゴール

HuggingFace Hub から `facebook/w2v-bert-2.0` モデルをダウンロードし、`Wav2Vec2BertModel.from_pretrained()` で正常にロードできる状態にする。セマンティックトークン抽出（`maskgct_utils.py` の `get_w2v2bert_tokens`）で使用される。

## 2. 実装する内容の詳細

```bash
huggingface-cli download facebook/w2v-bert-2.0
```

### ロード確認

```bash
uv run python -c "
from transformers import Wav2Vec2BertModel
model = Wav2Vec2BertModel.from_pretrained('facebook/w2v-bert-2.0')
print(f'OK: {sum(p.numel() for p in model.parameters()):,} parameters loaded')
"
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 環境構築担当 | 1 | ダウンロード実行、`from_pretrained` ロード確認 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: `facebook/w2v-bert-2.0` のダウンロードとロード確認

**含まないもの**: セマンティックトークン抽出（M1-13）、`wav2vec2bert_stats.pt`（M0-11）

### 4.2 ユニットテスト

```bash
uv run python -c "
from transformers import Wav2Vec2BertModel
model = Wav2Vec2BertModel.from_pretrained('facebook/w2v-bert-2.0')
assert model is not None
print('OK')
"
```

### 4.3 E2Eテスト

M1-13 のテストで音声ファイル入力からトークン抽出まで確認（本チケット範囲外）。

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- モデルサイズが大きい（推定 600MB〜1GB+）
- `transformers` バージョンによる API 差異

### 5.2 レビュー項目

- [ ] `from_pretrained` が警告なしで完了しているか
- [ ] `maskgct_utils.py` の参照モデル ID が `facebook/w2v-bert-2.0` であること確認
- [ ] revision SHA を記録したか

## 6. フェーズ振り返り: 一から作り直すとしたら

- `--local-dir` でプロジェクト内に保存、`from_pretrained` にはローカルパスを渡す
- `model.save_pretrained('./local_w2v_bert')` でローカル保存し完全オフライン動作を実現
- CI で `from_pretrained` → ダミー forward → 出力形状確認まで自動実行

## 7. 後続タスクへの連絡事項

- **M0-11**: `wav2vec2bert_stats.pt` は本モデルの出力正規化用。次元の整合確認が必要。
- **M1-13**: `maskgct_utils.py` の `get_w2v2bert_tokens` でのロードパターンを確認し引き継ぐ。
