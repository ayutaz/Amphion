# M0-11: wav2vec2bert_stats.pt 配置確認

> **マイルストーン**: [M0: 環境構築](../../13_milestones.md#m0-環境構築)
> **対応RQ**: -（環境構築）
> **依存チケット**: なし
> **ブロックするチケット**: M1-13
> **状態**: TODO

---

## 1. 目的とゴール

`models/tts/maskgct/ckpt/wav2vec2bert_stats.pt` が正しい形式で存在することを確認する。w2v-bert-2.0 の出力を平均/分散正規化するための統計情報（mean/var）を含み、`maskgct_utils.py` の `get_w2v2bert_tokens` で必須。

## 2. 実装する内容の詳細

### 受入確認

```bash
uv run python -c "
import torch
d = torch.load('models/tts/maskgct/ckpt/wav2vec2bert_stats.pt', map_location='cpu')
assert 'mean' in d and 'var' in d
print(f'mean shape: {d[\"mean\"].shape}, var shape: {d[\"var\"].shape}')
print('OK')
"
```

### ファイル取得方法（未存在時）

```bash
# 方法 A: HuggingFace から
huggingface-cli download amphion/MaskGCT \
  --include "wav2vec2bert_stats.pt" \
  --local-dir models/tts/maskgct/ckpt/

# 方法 B: sparse-checkout で既存ファイル確認
ls -la models/tts/maskgct/ckpt/
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 環境構築担当 | 1 | ファイル存在確認、形式検証、未存在時のダウンロード |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: `wav2vec2bert_stats.pt` の存在確認と `['mean', 'var']` キー検証

**含まないもの**: 統計値の数値的妥当性の詳細検証（M1-13）

### 4.2 ユニットテスト

```bash
uv run python -c "
import torch, os
path = 'models/tts/maskgct/ckpt/wav2vec2bert_stats.pt'
assert os.path.exists(path)
d = torch.load(path, map_location='cpu')
assert set(d.keys()) == {'mean', 'var'}
assert d['mean'].ndim == 1
print(f'OK: shape = {d[\"mean\"].shape}')
"
```

### 4.3 E2Eテスト

M1-13 のテストで正規化が正常動作することを確認（本チケット範囲外）。

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- `amphion/MaskGCT` リポジトリに含まれていない可能性（高）→ 自前計算で対応
- `torch.load` の `weights_only` オプション（PyTorch 2.4+）での読込失敗（低）

### 5.2 レビュー項目

- [ ] `d['mean'].shape[0]` が w2v-bert-2.0 出力次元と一致
- [ ] `maskgct_utils.py` のパスが `models/tts/maskgct/ckpt/wav2vec2bert_stats.pt` と一致
- [ ] ファイルの取得元を記録したか

## 6. フェーズ振り返り: 一から作り直すとしたら

- 統計ファイルをコードリポジトリに直接コミットする（数KB〜数MB）。HuggingFace 依存を排除。
- 生成時パラメータを `stats_metadata.json` として保存しバージョン管理する。
- Docker イメージに同梱する（小さいファイルのため）。

## 7. 後続タスクへの連絡事項

- **M1-13**: mean/var の shape（次元数 D）を引き継ぐこと。
- **M0-09**: w2v-bert-2.0 の出力次元と `wav2vec2bert_stats.pt` の次元が一致することを相互確認。
