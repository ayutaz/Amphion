# M3 共通設計判断

> このドキュメントは M3-01〜M3-22 の全チケットで共通する「一から作り直すとしたら」の設計判断をまとめたものです。
> 各チケットの Section 6 は「チケット固有の設計判断」のみを記載し、共通部分はこのドキュメントを参照してください。

---

## 1. PyTorch Lightning vs 素の Accelerate

本プロジェクトは Hugging Face **Accelerate** を採用している。

| 項目 | Accelerate（採用） | PyTorch Lightning（不採用） |
|---|---|---|
| DDP 設定 | `accelerator.prepare()` で統一 | `Trainer(strategy="ddp")` |
| LoRA 統合 | PEFT + Accelerate の事例が豊富 | Lightning + PEFT は設定が煩雑 |
| 学習ループ制御 | 明示的なループ（Algorithm 1 実装に適合） | `training_step` に隠蔽 |
| スモークテスト | `--max_steps 100` で代替 | `fast_dev_run=True` で内蔵 |
| チェックポイント | 手動の `accelerator.save_state()` | `ModelCheckpoint` コールバック自動管理 |

**採用理由**: Algorithm 1 の分岐ロジック（バッチ分割・ピッチ摂動・プロンプト生成）を明示的に実装するため、Accelerate のシンプルなラッパー構造の方が保守しやすい。

---

## 2. 実験管理（W&B / MLflow）

本プロジェクトは **Weights & Biases（W&B）** を推奨する。

- 損失の3成分（L_CL, L_SVT, L_mask）を個別に記録すること
- ハイパーパラメータ（lambda_scl, lambda_fcl, lambda_svt, lambda_mask, K_s, zero_prob 等）を run config として登録すること
- チェックポイントは W&B Artifact でバージョン管理することで、実験の再現性を保証する
- スモークテスト実験名には `smoke_` プレフィックスをつけ、本番実験と分離する

---

## 3. 学習率スケジューラ

S2A ファインチューニングには **inverse square root スケジューラ**（`get_scheduler("inverse_sqrt")`）を採用する。

```python
# HuggingFace transformers の get_scheduler を使用
from transformers import get_scheduler

scheduler = get_scheduler(
    "inverse_sqrt",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
)
```

- ウォームアップステップ数: `configs/comelsinger/s2a_train.yaml` の `scheduler.warmup_steps` を参照
- チェックポイント再開時はスケジューラの `state_dict` も必ず保存・復元すること

SVT 学習には `CosineAnnealingLR` を使用する（`svt_train.yaml` 参照）。

---

## 4. GradNorm 動的重み調整

現在の実装では損失重み（lambda_cl=0.5, lambda_svt=0.5, lambda_mask=0.3 等）は固定値。

将来的に GradNorm アルゴリズムを導入する場合は以下を検討すること：

- L_CL（対照学習）と L_mask（MaskGCT）の勾配スケールが大きく異なる場合に有効
- L_SVT の重みは学習フェーズ（初期 vs 後期）で最適値が変わる可能性がある
- 導入前に TensorBoard/W&B で各損失の grad norm を計測してから判断すること

---

## 5. 設定管理（Hydra / OmegaConf）

現在は `yaml.safe_load()` + argparse の構成。将来の改善として：

- Hydra + OmegaConf に移行すれば CLI オーバーライドが容易になる
- pydantic でスキーマ検証を追加し、型安全な設定管理を実現できる
- 設定ファイルのバージョニング（`version: "1.0"` フィールド）を追加することを推奨
