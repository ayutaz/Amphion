# M3-03: SVT学習スクリプト骨格

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07a
> **依存チケット**: M3-01
> **ブロックするチケット**: M3-04
> **状態**: TODO

---

## 1. 目的とゴール

`tools/train_svt.py` に argparse・設定読込・seed固定・オプティマイザとスケジューラの初期化を実装する。`--dry-run` フラグで0ステップ（学習ループ未実行）の状態で正常終了することを確認する。学習ループ本体は M3-04 で実装するため、骨格部分のみに集中する。

## 2. 実装する内容の詳細

```python
# tools/train_svt.py
import argparse, random, yaml
import numpy as np
import torch
import torch.optim as optim
from models.tts.comelsinger.svt_module import SVTModule

def parse_args():
    parser = argparse.ArgumentParser(description="SVT Training")
    parser.add_argument("--config", default="configs/comelsinger/svt_train.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["training"]["seed"])
    model = SVTModule().cuda()
    optimizer = optim.AdamW(model.parameters(),
        lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["scheduler"]["T_max"], eta_min=cfg["scheduler"]["eta_min"])
    if args.dry_run:
        print("dry-run: 0 steps OK"); return
    # 学習ループは M3-04 で実装
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | train_svt.py 骨格実装・dry-run動作確認 |

## 4. 提供範囲とテスト項目

**含むもの**: argparse設定、設定読込、seed固定、AdamW+CosineAnnealingLR初期化、--dry-run動作

**含まないもの**: 学習ループ（→ M3-04）、ロギング（→ M3-05）、バリデーション（→ M3-06）

### ユニットテスト

```bash
uv run python tools/train_svt.py --dry-run
# "dry-run: 0 steps OK" が出力されること
```

```bash
uv run python -c "
import subprocess, sys
result = subprocess.run([sys.executable, 'tools/train_svt.py', '--dry-run'],
    capture_output=True, text=True)
assert result.returncode == 0, result.stderr
assert 'dry-run' in result.stdout
print('PASS: dry-run exits normally')
"
```

### E2Eテスト

```bash
uv run python -c "
import subprocess, sys, yaml
with open('configs/comelsinger/svt_train.yaml') as f:
    cfg = yaml.safe_load(f)
result = subprocess.run(
    [sys.executable, 'tools/train_svt.py', '--config', 'configs/comelsinger/svt_train.yaml', '--dry-run'],
    capture_output=True, text=True)
assert result.returncode == 0
print('PASS: skeleton runs with config file')
"
```

## 5. 懸念事項とレビュー項目

- **GPUなし環境**: dry-runでのGPU依存を避けるため、`--device cpu` オプションを追加することを検討する。
- **設定ファイルパス**: 相対パスで指定した場合、実行ディレクトリによって解決が変わることに注意する。

### レビュー項目

- [ ] `--dry-run` で正常終了（exit code 0）すること
- [ ] seed=42が設定ファイルから読み込まれていること
- [ ] AdamW の lr が設定値と一致すること

## 6. フェーズ振り返り: 一から作り直すとしたら

> 共通の設計判断（PyTorch Lightning vs Accelerate、実験管理、スケジューラ選択）は [M3_design_decisions.md](M3_design_decisions.md) を参照のこと。以下はこのチケット固有の設計判断を記載する。


- **PyTorch Lightning vs 素のAccelerate**: LightningのTrainerを使えばdry-runは `fast_dev_run=True` で提供済み。骨格実装のコストが削減できる。
- **実験管理(W&B/MLflow)**: 骨格段階から `wandb.init()` か `mlflow.start_run()` を組み込むべき。後付けは困難。
- **設定管理**: Hydraを最初から使えばargparse不要で、設定のオーバーライドがCLIから可能。

## 7. 後続タスクへの連絡事項

- **M3-04**: `optimizer` と `scheduler` は本チケットで初期化済みのオブジェクトを引き継いで学習ループを実装すること。
- **M3-05**: TensorBoard の `SummaryWriter` 初期化も `main()` 内に追加する予定であることを念頭に置くこと。
- **M3-06**: `evaluate_svt()` 関数は別ファイル `tools/evaluate_svt.py` に実装し、`train_svt.py` から import する設計を推奨する。
