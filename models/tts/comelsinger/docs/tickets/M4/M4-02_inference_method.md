# M4-02: comelsinger_inference() メソッド実装

> **マイルストーン**: [M4: 推論・評価](../../13_milestones.md#m4-推論評価システム)
> **対応RQ**: RQ-08
> **依存チケット**: M4-01
> **ブロックするチケット**: M4-03
> **状態**: TODO

---

## 1. 目的とゴール

`CoMelSinger_Inference_Pipeline` に `comelsinger_inference()` メソッドを実装する。楽譜（歌詞＋ピッチ列）と参照音声を受け取り、`tokenize_score → T2S → S2A 第1層([25]) → S2A 全層([25,10,1,...]) → codec_decoder` の順で処理し、24kHz WAV を返す。このチケット完了時点でダミー入力による end-to-end の forward pass が通ること。

## 2. 実装する内容の詳細

```python
def comelsinger_inference(
    self,
    lyrics: list[str],           # 音素列（ピンイン）
    note_sequence: list[dict],   # [{pitch: int, duration: float}, ...]
    ref_audio: torch.Tensor,     # (1, T) 24kHz
    n_timesteps: list[int] = [25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
) -> np.ndarray:
    # Step 1: 楽譜トークン化
    phone_ids, pitch_ids, dur_ids = self.tokenize_score(lyrics, note_sequence)
    # Step 2: T2S でセマンティックトークン生成
    semantic_ids = self.t2s_model.inference(phone_ids, pitch_ids, dur_ids)
    # Step 3: S2A 第1層（n_timesteps[0]=25）
    acoustic_layer0 = self.s2a_model.inference_layer(semantic_ids, pitch_ids, layer=0,
                                                      n_timesteps=n_timesteps[0])
    # Step 4: S2A 全層（残り11層, RVQ 12層構成）
    acoustic_tokens = self.s2a_model.inference_all_layers(
        acoustic_layer0, semantic_ids, pitch_ids, n_timesteps=n_timesteps[1:])
    # Step 5: codec decoder
    waveform = self.codec_decoder(acoustic_tokens)
    return waveform.squeeze().cpu().numpy()
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `comelsinger_inference()` 本体実装 |
| デバッグエージェント | 1 | テンソル shape 追跡・デバイス整合性確認 |

## 4. 提供範囲とテスト項目

**含むもの**: `comelsinger_inference()` 完全実装、`tokenize_score()` ヘルパー

**含まないもの**: モデル重みロード（→ M4-03）、バッチ処理（→ M4-04）

### ユニットテスト

```python
def test_tokenize_score_output_shapes(pipe):
    lyrics = ["ni", "hao"]
    notes = [{"pitch": 60, "duration": 0.5}, {"pitch": 62, "duration": 0.5}]
    phone_ids, pitch_ids, dur_ids = pipe.tokenize_score(lyrics, notes)
    assert phone_ids.shape == pitch_ids.shape == dur_ids.shape

def test_inference_returns_numpy(pipe_with_weights):
    wav = pipe_with_weights.comelsinger_inference(
        lyrics=["ni"], note_sequence=[{"pitch": 60, "duration": 1.0}],
        ref_audio=torch.zeros(1, 24000))
    assert isinstance(wav, np.ndarray)
```

### E2Eテスト

```bash
uv run python -c "
from models.tts.comelsinger.comelsinger_inference import CoMelSinger_Inference_Pipeline
import torch, numpy as np
# ダミー入力で forward pass が通ることを確認
print('E2E forward pass OK')
"
```

## 5. 懸念事項とレビュー項目

- **S2A の layer-by-layer 推論 API**: MaskGCT_S2A が `inference_layer` / `inference_all_layers` を公開しているか確認が必要。
- **n_timesteps リストの長さ**: 要件定義書にて RVQ codebook 数は 12 層と確定している。`n_timesteps` は 12 要素（第1層 `[25]` + 残り11層 `[10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]`）が正しい。10 要素（RVQ 10 層相当）は誤りなので注意すること。

レビュー項目:
- [ ] `tokenize_score` が空の歌詞・ノートでも例外を出さないか
- [ ] `ref_audio` のサンプリングレートが 24kHz でない場合のリサンプリング処理があるか
- [ ] `n_timesteps` のデフォルト値が論文 Table IV と一致しているか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **Gradio/Streamlit UI**: 推論メソッドを設計する際、`gr.Interface` でラップしやすい入出力型（`str, str, np.ndarray` など）を最初から意識する。
- **バッチ推論最適化**: 逐次 forward を前提にすると、複数曲を並列推論できない。`comelsinger_batch_inference()` を最初から用意し、DataLoader で動かせるようにすべき。
- **推論サーバ化**: FastAPI エンドポイントとして切り出せるよう、シリアライズ可能な入出力型に統一する。

## 7. 後続タスクへの連絡事項

- M4-03 は本メソッドが完成した後で LoRA 重みをロードする層を追加する。`comelsinger_inference()` 自体は重みロードに依存しないが、M4-03 完了前は正確な出力は得られない。
- M4-04 の `run_inference.py` は `comelsinger_inference()` を直接呼び出す。シグネチャを変更した場合は M4-04 担当者に通知すること。
- `tokenize_score()` の入力フォーマット（歌詞・ノート列の型定義）を M4-04 担当者に文書化して渡すこと。
