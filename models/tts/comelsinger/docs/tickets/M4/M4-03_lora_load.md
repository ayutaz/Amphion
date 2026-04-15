# M4-03: LoRA チェックポイントロードと pitch_emb ロード

> **マイルストーン**: [M4: 推論・評価](../../13_milestones.md#m4-推論評価システム)
> **対応RQ**: RQ-08
> **依存チケット**: M4-02
> **ブロックするチケット**: M4-04
> **状態**: TODO

---

## 1. 目的とゴール

学習済みの LoRA チェックポイント（`s2a_lora.pt`）および pitch embedding 重み（`pitch_emb.pt`）を `CoMelSinger_Inference_Pipeline` にロードするメソッドを実装する。1-layer モデル（RVQ 第1層のみ）と full モデル（全8層）の両方に対応し、ロード後に推論が正常に動作することを確認する。

## 2. 実装する内容の詳細

```python
from peft import PeftModel

def load_lora_checkpoint(
    self,
    lora_ckpt_dir: str,       # LoRA アダプター保存ディレクトリ
    pitch_emb_path: str,      # pitch_emb.pt のパス
    model_mode: str = "full", # "1layer" or "full"
) -> None:
    # LoRA アダプターをロード
    self.s2a_model.backbone = PeftModel.from_pretrained(
        self.s2a_model.backbone,
        lora_ckpt_dir,
        is_trainable=False,
    )
    # pitch_emb をロード
    state = torch.load(pitch_emb_path, map_location=self.device)
    self.s2a_model.pitch_emb.load_state_dict(state)
    # モデルモードを設定
    self.model_mode = model_mode
    self.s2a_model.eval()
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `load_lora_checkpoint()` 実装・PEFT 連携 |
| 検証エージェント | 1 | ロード後の重みサイズ・dtype 確認 |

## 4. 提供範囲とテスト項目

**含むもの**: `load_lora_checkpoint()` メソッド、1layer/full モード切り替え

**含まないもの**: LoRA 学習自体（M3 の範囲）、チェックポイントのフォーマット変換

### ユニットテスト

```python
def test_lora_load_sets_eval_mode(pipe, tmp_lora_dir):
    pipe.load_lora_checkpoint(tmp_lora_dir, "pitch_emb.pt", model_mode="full")
    assert not pipe.s2a_model.backbone.training

def test_pitch_emb_weights_loaded(pipe, tmp_lora_dir):
    pipe.load_lora_checkpoint(tmp_lora_dir, "pitch_emb.pt")
    # pitch_emb の requires_grad は False（推論モード）
    for p in pipe.s2a_model.pitch_emb.parameters():
        assert not p.requires_grad
```

### E2Eテスト

```bash
uv run python -c "
from models.tts.comelsinger.comelsinger_inference import CoMelSinger_Inference_Pipeline
pipe = CoMelSinger_Inference_Pipeline(cfg, device='cpu')
pipe.load_lora_checkpoint('checkpoints/lora', 'checkpoints/pitch_emb.pt')
print('LoRA load OK, mode:', pipe.model_mode)
"
```

## 5. 懸念事項とレビュー項目

- **PEFT バージョン依存**: `PeftModel.from_pretrained` の API は PEFT バージョンで変わる場合がある。`pyproject.toml` でバージョンをピン留めすること。
- **dtype 不一致**: LoRA 重みが `float16` で保存されている場合、ベースモデルの `float32` と混在する。`.to(self.s2a_model.dtype)` で統一すること。

レビュー項目:
- [ ] `model_mode="1layer"` 時に全8層の LoRA がロードされないか（意図的な部分ロードになっているか）
- [ ] `pitch_emb.pt` が見つからない場合に分かりやすいエラーメッセージが出るか
- [ ] `.eval()` の呼び出しが LoRA アダプターにも適用されるか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **Gradio/Streamlit UI**: チェックポイントパスをファイルダイアログで指定できるよう、ロードメソッドを UI 呼び出し可能な形にする。
- **バッチ推論最適化**: LoRA ロードを複数のワーカープロセスで共有するには、`torch.multiprocessing` の shared memory か ONNX Export を検討すべき。
- **推論サーバ化**: LoRA 重みをサーバ起動時に一度だけロードし、リクエスト間で再利用できるシングルトンパターンを採用する。

## 7. 後続タスクへの連絡事項

- M4-04 は `load_lora_checkpoint()` を CLI から呼び出す。チェックポイントパスを `--lora_ckpt_dir` 引数として渡す設計を M4-04 に伝えること。
- M4-10 の ablation では LoRA なし（ベースモデルのみ）の実行も必要。`load_lora_checkpoint` をスキップできるオプションを M4-04 の CLI に追加すること。
- LoRA チェックポイントのディレクトリ構造（`adapter_config.json`, `adapter_model.safetensors`）を M4-04 担当者に文書化して渡すこと。
