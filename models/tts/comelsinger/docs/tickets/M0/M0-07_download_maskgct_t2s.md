# M0-07: 事前学習済みモデル: MaskGCT-T2S

> **マイルストーン**: [M0: 環境構築](../../13_milestones.md#m0-環境構築)
> **対応RQ**: -（環境構築）
> **依存チケット**: なし
> **ブロックするチケット**: M3-03（T2S推論で使用）
> **状態**: TODO

---

## 1. 目的とゴール

HuggingFace Hub から `amphion/MaskGCT-T2S` モデルの重みファイルをダウンロードし、ローカルキャッシュに配置する。T2S 推論パイプラインで使用する初期重みを確保する。

## 2. 実装する内容の詳細

```bash
huggingface-cli download amphion/MaskGCT-T2S

# ダウンロード対象ファイルの事前確認
huggingface-cli repo-files amphion/MaskGCT-T2S
```

### ファイルサイズ検証

```python
from huggingface_hub import scan_cache_dir
cache_info = scan_cache_dir()
t2s_repos = [r for r in cache_info.repos if r.repo_id == "amphion/MaskGCT-T2S"]
assert len(t2s_repos) > 0, "amphion/MaskGCT-T2S not found in cache"
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 環境構築担当 | 1 | ダウンロード実行、ファイルサイズ確認 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: `amphion/MaskGCT-T2S` の全ファイルをローカルキャッシュにダウンロード

**含まないもの**: `load_state_dict` による重み読込検証（M0-12）、ローカルディレクトリへのコピー

### 4.2 ユニットテスト

```bash
uv run python -c "
from huggingface_hub import scan_cache_dir
cache = scan_cache_dir()
repos = [r for r in cache.repos if r.repo_id == 'amphion/MaskGCT-T2S']
assert len(repos) == 1, 'MaskGCT-T2S not cached'
print('OK')
"
```

### 4.3 E2Eテスト

M0-12 の smoke test でモデルロードを含めた E2E 確認を実施。

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- ネットワーク帯域・ダウンロード時間（モデルサイズ数GB）
- HuggingFace のリポジトリ構造変更リスク
- アクセストークン未設定時の 403 エラー

### 5.2 レビュー項目

- [ ] キャッシュパスが `maskgct_inference.py` の読み込みパスと一致しているか
- [ ] revision SHA を記録したか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **モデルダウンロード戦略**: `--local-dir models/tts/maskgct/ckpt/t2s/` でプロジェクト内へ直接保存する。HuggingFace キャッシュは環境依存でパスが変わり再現性に難がある。
- **モデルバージョン管理**: `--revision <SHA>` で特定コミットを指定し `pyproject.toml` に記録。DVC でモデルファイルのバージョン管理も検討。
- **Docker 化**: `Dockerfile` に `RUN huggingface-cli download ...` を記述。
- **ネットワーク非依存**: `HF_HUB_OFFLINE=1` モード対応。Git LFS/DVC でリポジトリに含めるオプションを設ける。

## 7. 後続タスクへの連絡事項

- **M0-12**: キャッシュパスまたはローカルパスを `from_pretrained` 引数として渡す方法を確認すること。
- revision SHA を `13_milestones.md` に記録しておくこと。
