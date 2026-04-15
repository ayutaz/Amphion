# M0-08: 事前学習済みモデル: MaskGCT-S2A

> **マイルストーン**: [M0: 環境構築](../../13_milestones.md#m0-環境構築)
> **対応RQ**: -（環境構築）
> **依存チケット**: なし
> **ブロックするチケット**: M3-08（S2A LoRA fine-tuning初期重み）
> **状態**: TODO

---

## 1. 目的とゴール

HuggingFace Hub から `amphion/MaskGCT-S2A` モデルの重みファイルをダウンロードし、`load_state_dict` でエラーなく読み込める状態にする。CoMelSinger S2A の LoRA fine-tuning における初期重みとして使用する。

**ゴール**: 1-layer モデルと full モデルの両方がキャッシュに存在し、`unexpected keys` なしでロード可能。

## 2. 実装する内容の詳細

```bash
huggingface-cli download amphion/MaskGCT-S2A

# ファイル構成確認
huggingface-cli repo-files amphion/MaskGCT-S2A
```

### 2モデル構成

- **1-layer モデル**: 反復デコーディングの初期ステップ（高速・低品質）
- **full モデル**: 最終ステップ（低速・高品質）

CoMelSinger の LoRA fine-tuning は **両方のモデル** を対象とする。

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 環境構築担当 | 1 | ダウンロード実行、ファイル構成確認、load_state_dict 検証 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: 1-layer + full 両モデルのダウンロード、`load_state_dict` 確認

**含まないもの**: LoRA 適用（M2-15）、S2A 推論実行（M4）

### 4.2 ユニットテスト

```bash
uv run python -c "
from huggingface_hub import scan_cache_dir
cache = scan_cache_dir()
repos = [r for r in cache.repos if r.repo_id == 'amphion/MaskGCT-S2A']
assert len(repos) == 1, 'MaskGCT-S2A not cached'
print('OK')
"
```

### 4.3 E2Eテスト

M0-12 の smoke test に統合。

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- `amphion/MaskGCT-S2A` のファイル構成が公開ドキュメントに未記載（高）→ `repo-files` で事前確認
- 1-layer と full の state_dict キー名の差異（中）→ `strict=False` で確認
- LoRA 拡張後の key 不整合（中）→ M2-15 で事前記録

### 5.2 レビュー項目

- [ ] 1-layer / full 両方がダウンロード済みか
- [ ] `missing_keys` が LoRA 追加予定のパラメータのみか
- [ ] revision SHA を記録したか

## 6. フェーズ振り返り: 一から作り直すとしたら

- `--local-dir` でプロジェクト内に保存しキャッシュ依存を排除
- `--revision <SHA>` で固定、`pyproject.toml` に記録
- DVC で S2A モデルを管理し `dvc pull` で再現可能にする

## 7. 後続タスクへの連絡事項

- **M3-08**: LoRA fine-tuning の初期重みパスを引き継ぐ。特に full モデルのパスを `train_s2a.py` 設定に記載。
- **M2-15**: `get_peft_model()` 適用前後の key 差分を事前記録しておくこと。
