# M0-10: 事前学習済みモデル: Amphion Codec

> **マイルストーン**: [M0: 環境構築](../../13_milestones.md#m0-環境構築)
> **対応RQ**: -（環境構築）
> **依存チケット**: M0-01
> **ブロックするチケット**: M1-12
> **状態**: TODO

---

## 1. 目的とゴール

`amphion/MaskGCT` HuggingFace リポジトリから `CodecEncoder`/`CodecDecoder` の事前学習済み重みをダウンロードし、設定ファイルとの整合性を確認する。音響トークン抽出および推論時の波形生成に使用。

## 2. 実装する内容の詳細

### ダウンロード元の特定

```bash
# maskgct_utils.py の codec ロードパターンを確認
grep -n "codec" models/tts/maskgct/maskgct_utils.py | head -30
grep -n "codec" models/tts/maskgct/maskgct_inference.py | head -30
```

### ダウンロード

```bash
huggingface-cli download amphion/MaskGCT \
  --include "*.safetensors" "*.bin" "*.json" \
  --local-dir models/tts/maskgct/ckpt/
```

**注意**: ファイルパスは `maskgct_inference.py` のロードコードで確認してから実行。

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 環境構築担当 | 1 | ダウンロード元特定、実行、設定整合性確認 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: Codec 重みファイルのダウンロード・配置、アーキテクチャ整合性確認

**含まないもの**: 音響トークン抽出実行（M1-12）、RVQ デコード（M4）

### 4.2 ユニットテスト

```bash
uv run python -c "
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
enc = CodecEncoder()
dec = CodecDecoder()
print(f'CodecEncoder params: {sum(p.numel() for p in enc.parameters()):,}')
print('OK')
"
```

### 4.3 E2Eテスト

M1-12 のテストで実際の音声ファイルから RVQ トークン列を取得して確認（本チケット範囲外）。

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- `amphion/MaskGCT` の Codec 重みのファイル名・構造が不明確（高）
- RVQ コードブック数の設定と重みの不整合（高）
- `CodecEncoder/Decoder` の `__init__` 引数がデフォルト値でない可能性（中）

### 5.2 レビュー項目

- [ ] ダウンロード元のリポジトリ ID + ファイルパスを記録したか
- [ ] `load_state_dict` で `unexpected_keys` が空か
- [ ] RVQ コードブック数が **12** であることを重みで確認したか（注: CLAUDE.md の図では「RVQ 8層」と記載されているが、要件定義書12で `num_quantizer=12` に修正済み。MaskGCT ソースコードのデフォルト値も12）

## 6. フェーズ振り返り: 一から作り直すとしたら

- `--local-dir models/tts/maskgct/ckpt/codec/` で明示的に保存
- `--revision <SHA>` で固定、モデルと実装コードのバージョン対応表を作成
- DVC で `models/tts/maskgct/ckpt/` 以下の重みを管理

## 7. 後続タスクへの連絡事項

- **M1-12**: Codec 重みのローカルパスを `extract_acoustic_tokens` の設定として引き渡す。
- **M0-12**: RVQ 層数・コードブックサイズを連絡すること。
