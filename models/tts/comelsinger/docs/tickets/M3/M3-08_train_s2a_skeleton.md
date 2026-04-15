# M3-08: S2A学習スクリプト骨格

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07b
> **依存チケット**: M3-07
> **ブロックするチケット**: M3-09
> **状態**: TODO

---

## 1. 目的とゴール

`tools/train_s2a.py` にAccelerate（bf16、DDP、find_unused_parameters=True）設定、CoMelSinger_S2Aモデル構築、MaskGCT事前学習重みのstrict=Falseロード、LoRA適用、frozen SVTモデルの読込、AdamW＋逆平方根スケジューラの初期化を実装する。モデルが正しく構築されてforwardが通ることを確認できる骨格を作る。

## 2. 実装する内容の詳細

```python
# tools/train_s2a.py
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
from models.tts.comelsinger.svt_module import SVTModule
import torch

def build_model_and_optimizer(cfg, accelerator):
    # S2A モデル構築 + 事前学習重みロード
    model = CoMelSinger_S2A(**cfg["model"])
    ckpt = torch.load(cfg["pretrained"]["s2a_path"], map_location="cpu")
    model.load_state_dict(ckpt, strict=False)

    # LoRA 適用
    lora_cfg = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
        target_modules=cfg["lora"]["target_modules"])
    model = get_peft_model(model, lora_cfg)

    # SVT frozen ロード
    svt = SVTModule(); svt.load_state_dict(torch.load(cfg["svt_checkpoint"]))
    for p in svt.parameters(): p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["optimizer"]["lr"])
    return model, svt, optimizer
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | Accelerate設定・LoRA適用・SVT frozen・スケジューラ初期化 |

## 4. 提供範囲とテスト項目

**含むもの**: Accelerate初期化、CoMelSinger_S2A構築、事前学習重みstrict=Falseロード、LoRA適用、frozen SVT

**含まないもの**: 学習ループ（→ M3-09〜M3-18）、ロギング（→ M3-19）

### ユニットテスト

```bash
uv run python -c "
from peft import LoraConfig
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'])
print(f'PASS: LoraConfig r={cfg.r}, alpha={cfg.lora_alpha}')
"
```

```bash
uv run python -c "
from accelerate import Accelerator
acc = Accelerator(mixed_precision='bf16',
    kwargs_handlers=[{'find_unused_parameters': True}])
print(f'PASS: Accelerator device={acc.device}, precision={acc.mixed_precision}')
"
```

### E2Eテスト

```bash
uv run python tools/train_s2a.py \
    --config configs/comelsinger/s2a_train.yaml \
    --dry-run
# モデル構築完了・LoRAパラメータ数表示・exit 0
```

## 5. 懸念事項とレビュー項目

- **find_unused_parameters=True**: frozen SVTの存在によりDDP backward時に未使用パラメータが発生するため必須。
- **strict=False**: 事前学習重みにないピッチ埋め込み層のキーを無視するために必要。ロード時の不一致キーをログ出力する。

### レビュー項目

- [ ] LoRAがq_proj, v_projにのみ適用されていること
- [ ] SVTの全パラメータがrequires_grad=Falseであること
- [ ] 事前学習重みのロード時に不足・余剰キーがログ出力されること

## 6. フェーズ振り返り: 一から作り直すとしたら

- **PyTorch Lightning vs 素のAccelerate**: LightningはDDP設定がシンプルで `find_unused_parameters` は自動検出できる場合がある。ただしLoRAとの統合は Accelerate + PEFT の方が事例が多い。
- **実験管理(W&B/MLflow)**: モデル構築時のパラメータ数（LoRA有効/全体）をW&Bに記録する。LoRAが約4.8%に設定通りか確認できる。
- **学習率スケジューラ選択**: 逆平方根スケジューラ（InverseSquareRoot）はHuggingFace transformersの `get_scheduler("inverse_sqrt")` で実装可能。ウォームアップステップ数の設定が重要。

## 7. 後続タスクへの連絡事項

- **M3-09**: `model`, `svt`, `optimizer` は `build_model_and_optimizer()` から返されるオブジェクトをそのまま学習ループで使用する。
- **M3-12**: `cond_emb` はCoMelSinger_S2Aの前半エンコーダ出力を指す。forward時の引数設計を事前に確認すること。
- **M3-17**: frozen SVTの `torch.no_grad()` コンテキストはM3-17で実装するが、本チケットでSVTの `requires_grad=False` が正しく設定されていることが前提条件となる。
