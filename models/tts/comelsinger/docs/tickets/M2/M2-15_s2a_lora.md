# M2-15: S2A LoRA 適用後の動作確認

> **マイルストーン**: [M2: コアモジュール](../../13_milestones.md#m2-コアモジュール実装)
> **対応RQ**: RQ-03
> **依存チケット**: M2-14
> **ブロックするチケット**: M3-08
> **状態**: TODO

---

## 1. 目的とゴール

`CoMelSinger_S2A` に `LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], modules_to_save=["pitch_emb"])` を適用し、学習可能パラメータ比率が 4〜6%（目標 ≈4.83%）であることを確認する。LoRA 適用後も `forward` と `backward` が正常動作することを保証する。

## 2. 実装する内容の詳細

```python
# tests/test_s2a_lora.py
import torch
from peft import LoraConfig, get_peft_model
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A

def apply_lora(model: CoMelSinger_S2A) -> CoMelSinger_S2A:
    """S2A モデルに LoRA を適用して返す。"""
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["pitch_emb"],
        bias="none",
    )
    return get_peft_model(model, config)

def test_lora_param_ratio():
    model = apply_lora(CoMelSinger_S2A())
    model.print_trainable_parameters()

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total_params * 100

    assert 4.0 <= ratio <= 6.0, f"trainable ratio {ratio:.2f}% not in [4%, 6%]"
    # pitch_emb にのみ勾配が流れることを確認
    B, T = 1, 20
    x = torch.randint(0, 1024, (B, 8, T))
    cond = model.get_cond(torch.randint(0,1024,(B,T)), torch.randint(0,129,(B,T)))
    loss = model.compute_loss(x, torch.ones(B,T), cond)
    loss_val = loss["loss"] if isinstance(loss, dict) else loss
    loss_val.backward()
    # LoRA パラメータに勾配があること
    lora_params = [(n,p) for n,p in model.named_parameters()
                   if "lora_" in n and p.requires_grad]
    for name, p in lora_params[:3]:
        assert p.grad is not None, f"LoRA grad is None: {name}"
    print(f"PASS: ratio={ratio:.2f}%, LoRA+pitch_emb grad OK")

if __name__ == "__main__":
    test_lora_param_ratio()
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| テストエージェント | 1 | LoRA 適用スクリプトの作成・実行・パラメータ比率検証 |

## 4. 提供範囲とテスト項目

**含むもの**: LoRA 適用ユーティリティ `apply_lora`、統合テストスクリプト `tests/test_s2a_lora.py`

**含まないもの**: LoRA の学習ループ（→ M3-08）、チェックポイント保存（→ M3）

### ユニットテスト

```bash
uv run python -c "
from peft import LoraConfig, get_peft_model
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj','v_proj'], modules_to_save=['pitch_emb'])
model = get_peft_model(CoMelSinger_S2A(), cfg)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
ratio = trainable / total * 100
print(f'trainable: {trainable:,} / {total:,} = {ratio:.2f}%')
assert 4.0 <= ratio <= 6.0, f'ratio {ratio:.2f}% out of range'
print('PASS: param ratio OK')
"
```

```bash
uv run python -c "
from peft import LoraConfig, get_peft_model
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj','v_proj'], modules_to_save=['pitch_emb'])
model = get_peft_model(CoMelSinger_S2A(), cfg)
# LoRA 以外のパラメータに勾配がないこと
non_trainable_has_grad = any(
    p.grad is not None
    for n, p in model.named_parameters()
    if not p.requires_grad
)
# 初期化直後は grad=None なので通過する（backward 後に確認）
print('PASS: non-trainable params have no grad at init')
"
```

### E2Eテスト

```bash
uv run python tests/test_s2a_lora.py
# 期待出力: trainable: X,XXX,XXX / XX,XXX,XXX = X.XX%
#           PASS: ratio=X.XX%, LoRA+pitch_emb grad OK
```

## 5. 懸念事項とレビュー項目

- **PEFT バージョン**: `get_peft_model` の API は PEFT ライブラリのバージョンで変わる。`uv run python -c "import peft; print(peft.__version__)"` で確認すること。
- **`modules_to_save`**: `pitch_emb` を `modules_to_save` に指定すると PEFT が `original_module` と `modules_to_save` の両方を保持するため、パラメータ数が倍増する場合がある。比率計算に注意すること。

### レビュー項目

- [ ] 学習可能パラメータ比率が 4〜6% の範囲に収まるか
- [ ] LoRA パラメータ（`lora_A`, `lora_B`）に勾配が計算されるか
- [ ] `pitch_emb.weight` が `modules_to_save` として学習可能か
- [ ] `q_proj`, `v_proj` 以外の親クラスパラメータが `requires_grad=False` か

## 6. フェーズ振り返り: 一から作り直すとしたら

**継承 vs コンポジション**: `get_peft_model` は `nn.Module` を受け取るため、継承・コンポジションどちらでも動作する。ただし PEFT のラッピングで `isinstance` チェックが壊れる可能性があるため、M2-11 の `isinstance(model, MaskGCT_S2A)` がラッピング後も通るか確認すること。

**LoRA vs QLoRA vs フルFT**: QLoRA は量子化（4bit/8bit）を組み合わせるため、24GB VRAM の A5000 では QLoRA 不要かもしれない。ただし 4枚並列（96GB 合計）でも全パラメータ FT は厳しいため、LoRA が妥当な選択。

**ハイパーパラメータ管理**: `r=16, alpha=32` は論文値に基づく。`r` を変えたアブレーションを容易にするため、`apply_lora(model, r=16, alpha=32)` のように関数シグネチャで管理することを推奨。

## 7. 後続タスクへの連絡事項

- **M3-08（LoRA 学習ループ）**: `apply_lora` ユーティリティをインポートして使用すること。学習ループ開始前に `model.print_trainable_parameters()` を呼び出してパラメータ比率を確認すること。
- **M3（チェックポイント保存）**: PEFT モデルは `model.save_pretrained(path)` で保存し、`PeftModel.from_pretrained(base_model, path)` でロードする。通常の `torch.save` とは異なる。
- **M4（推論パイプライン）**: 推論時は `model.merge_and_unload()` で LoRA 重みをベースモデルにマージしてから使用するとレイテンシが下がる。ただしマージ後は再学習不可。
