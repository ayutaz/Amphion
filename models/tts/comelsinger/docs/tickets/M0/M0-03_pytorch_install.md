# M0-03: PyTorch 系ライブラリインストール

> **マイルストーン**: [M0: 環境構築](../../13_milestones.md#m0-環境構築)
> **対応RQ**: -（環境構築）
> **依存チケット**: M0-02
> **ブロックするチケット**: M0-04, M0-05, M0-06, M0-12
> **状態**: TODO

---

## 1. 目的とゴール

CoMelSinger の学習・推論はすべて PyTorch を基盤とする。`torch` と `torchaudio` がインストールされていない状態では、後続の ML 系ライブラリもインポートできない。

このチケットを完了すると `torch>=2.2.0` と `torchaudio>=2.2.0` がインストールされ、CUDA デバイスが認識される状態になる。

## 2. 実装する内容の詳細

### 手順（CPU 環境 / 確認用）

```bash
uv add "torch>=2.2.0" "torchaudio>=2.2.0"
```

### 手順（CUDA 12.1 環境 / 本番 GPU サーバ）

```bash
uv add "torch>=2.2.0" "torchaudio>=2.2.0" \
  --index-url https://download.pytorch.org/whl/cu121
```

`pyproject.toml` に以下を追記して永続化する:

```toml
[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu121" }
torchaudio = { index = "pytorch-cu121" }
```

### CUDA バージョンの確認

```bash
nvidia-smi
uv run python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('GPU count:', torch.cuda.device_count())
"
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実行エージェント | 1 | uv add 実行・CUDA 確認・動作テスト |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: `torch>=2.2.0`, `torchaudio>=2.2.0` のインストール、GPU 認識確認

**含まないもの**: NVIDIA ドライバ・CUDA ツールキットのインストール、`torchvision`（本プロジェクトでは不要）

### 4.2 ユニットテスト

```bash
uv run python -c "
import torch, torchaudio
assert torch.__version__ >= '2.2.0'
assert torchaudio.__version__ >= '2.2.0'
print('PASS')
"
```

### 4.3 E2Eテスト

```bash
uv run python -c "
import torch
x = torch.randn(3, 3).cuda()
y = torch.randn(3, 3).cuda()
z = x @ y
print('GPU tensor matmul:', z.device)
print('PASS')
"
```

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- **CUDA バージョン不一致**: `nvidia-smi` の CUDA バージョンと PyTorch ビルドの不一致で `torch.cuda.is_available()` が `False` になる。
- **macOS での CPU ビルド**: 開発用 macOS では `torch.cuda.is_available()` は `False`。GPU 確認は本番サーバのみ。

### 5.2 レビュー項目

- [ ] `torch.__version__` が `2.2.0` 以上か
- [ ] GPU サーバ上で `torch.cuda.is_available()` が `True` か
- [ ] `pyproject.toml` にインデックス設定が記録されているか

## 6. フェーズ振り返り: 一から作り直すとしたら

**PyTorch インデックス管理**: uv の `[[tool.uv.index]]` で PyTorch インデックスを設定し `uv.lock` に固定する。一から設計するなら CUDA バージョンを `.env` で管理し CI/CD で自動選択する。

**モデルキャッシュ戦略**: HuggingFace のキャッシュは `HF_HOME` 環境変数で共有ディレクトリを指定し、NFS で全ノードが同一キャッシュを参照することでダウンロードの重複を防ぐ。

## 7. 後続タスクへの連絡事項

- **M0-04〜M0-06**: `torch` がインストール済みであることが前提。
- `torch.__version__` の正確な値を後続に伝えること（`transformers` のバージョン依存性のため）。
- macOS では `torch.cuda.is_available()` は `False` だが正常。
