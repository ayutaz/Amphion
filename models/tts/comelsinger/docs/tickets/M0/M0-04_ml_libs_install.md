# M0-04: ML 系ライブラリインストール

> **マイルストーン**: [M0: 環境構築](../../13_milestones.md#m0-環境構築)
> **対応RQ**: -（環境構築）
> **依存チケット**: M0-03
> **ブロックするチケット**: M0-12
> **状態**: TODO

---

## 1. 目的とゴール

CoMelSinger の S2A 拡張は `transformers` の `LlamaConfig`/`LlamaModel` をベースとし、`peft` の LoRA でファインチューニングを行い、`accelerate` でマルチ GPU 学習を実現する。

このチケットを完了すると `transformers>=4.40.0`, `peft>=0.11.0`, `accelerate>=0.30.0` がインストールされる。

## 2. 実装する内容の詳細

```bash
uv add "transformers>=4.40.0" "peft>=0.11.0" "accelerate>=0.30.0"
```

### 確認

```bash
uv run python -c "
from peft import LoraConfig
from transformers import LlamaConfig, LlamaModel
print('PASS')
"
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実行エージェント | 1 | uv add 実行・インポート確認 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: `transformers>=4.40.0`, `peft>=0.11.0`, `accelerate>=0.30.0` のインストール

**含まないもの**: `datasets`, `bitsandbytes`, HuggingFace 認証設定

### 4.2 ユニットテスト

```bash
uv run python -c "
from peft import LoraConfig
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'])
assert cfg.r == 16
from transformers import LlamaConfig
llama_cfg = LlamaConfig(hidden_size=512, num_hidden_layers=4, num_attention_heads=8)
assert llama_cfg.hidden_size == 512
print('PASS')
"
```

### 4.3 E2Eテスト

```bash
uv run python -c "
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
print('LlamaDecoderLayer available:', LlamaDecoderLayer)
print('PASS')
"
```

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- **transformers の API 変更**: MaskGCT の `llama_nar.py` がプライベート API を使用している場合、バージョン更新で壊れる可能性がある。
- **peft と transformers のバージョン互換性**: `uv add` が自動解決するが `uv.lock` で確認すること。

### 5.2 レビュー項目

- [ ] `transformers.__version__` >= `4.40.0`
- [ ] `peft.__version__` >= `0.11.0`
- [ ] `from transformers.models.llama.modeling_llama import LlamaDecoderLayer` 成功

## 6. フェーズ振り返り: 一から作り直すとしたら

**LoRA アダプタの設計選択**: `peft` は HuggingFace 標準で保存・ロードが容易。`merge_and_unload()` で推論オーバーヘッドをゼロにできる。`target_modules` の選択（`q_proj, v_proj` のみ）が最適かは ablation で確認すべき。

**CI/CD での再現性**: `uv sync --frozen` + GitHub Actions キャッシュでインストール時間を大幅短縮可能。

## 7. 後続タスクへの連絡事項

- **M0-12**: `from transformers.models.llama.modeling_llama import LlamaDecoderLayer` が内部的に呼ばれる。
- `transformers.__version__` を記録し M1 以降の API 互換性の基準とすること。
- `accelerate config` による DDP 設定は M3-01 のスコープ。
