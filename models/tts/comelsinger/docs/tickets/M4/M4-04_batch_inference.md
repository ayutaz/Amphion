# M4-04: バッチ推論 CLI スクリプト (run_inference.py)

> **マイルストーン**: [M4: 推論・評価](../../13_milestones.md#m4-推論評価システム)
> **対応RQ**: RQ-08
> **依存チケット**: M4-03
> **ブロックするチケット**: M4-05, M4-06, M4-07, M4-08, M4-09
> **状態**: TODO

---

## 1. 目的とゴール

`run_inference.py` CLI スクリプトを実装し、Seen/Unseen 各 50 発話の一括推論を自動化する。`--config`, `--testset`, `--output_dir` の 3 引数を受け取り、全発話の WAV ファイルを `output_dir/` に書き出す。後続の評価チケット（M4-05〜M4-09）はこのスクリプトの出力を入力として使用する。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/run_inference.py
import argparse, json, soundfile as sf
from pathlib import Path
from models.tts.comelsinger.comelsinger_inference import CoMelSinger_Inference_Pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)   # YAML 設定ファイル
    parser.add_argument("--testset", required=True)  # JSON: [{id, lyrics, notes, ref_audio}, ...]
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora_ckpt_dir", default=None)
    parser.add_argument("--split", choices=["seen", "unseen", "all"], default="all")
    args = parser.parse_args()

    pipe = CoMelSinger_Inference_Pipeline.from_config(args.config)
    if args.lora_ckpt_dir:
        pipe.load_lora_checkpoint(args.lora_ckpt_dir, f"{args.lora_ckpt_dir}/pitch_emb.pt")

    samples = json.load(open(args.testset))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for s in samples:
        wav = pipe.comelsinger_inference(s["lyrics"], s["notes"], s["ref_audio_path"])
        sf.write(out / f"{s['id']}.wav", wav, 24000)
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | CLI スクリプト本体・引数パーサ |
| テストエージェント | 1 | Seen/Unseen 50 発話の完走確認・出力 WAV 検証 |

## 4. 提供範囲とテスト項目

**含むもの**: `run_inference.py` CLI、`--split` フィルタ、WAV 書き出し

**含まないもの**: 評価指標計算（M4-05〜M4-09）、ablation 条件切り替え（M4-10）

### ユニットテスト

```python
def test_cli_creates_output_dir(tmp_path, cfg_path, testset_path):
    subprocess.run(["uv", "run", "python", "run_inference.py",
                    "--config", cfg_path, "--testset", testset_path,
                    "--output_dir", str(tmp_path / "out")], check=True)
    assert (tmp_path / "out").exists()

def test_output_wav_count(tmp_path, cfg_path, testset_50_path):
    # 50 発話 → 50 ファイル
    run_cli(tmp_path, cfg_path, testset_50_path)
    assert len(list((tmp_path / "out").glob("*.wav"))) == 50
```

### E2Eテスト

```bash
uv run python models/tts/comelsinger/run_inference.py \
  --config configs/comelsinger.yaml \
  --testset data/testset_seen.json \
  --output_dir outputs/seen_results \
  --split seen
ls outputs/seen_results/*.wav | wc -l  # → 50
```

## 5. 懸念事項とレビュー項目

- **GPU メモリ**: 50 発話を逐次処理しても VRAM が解放されない場合がある。ループ内で `torch.cuda.empty_cache()` を呼ぶ。
- **長曲のタイムアウト**: 1 発話あたりの最大推論時間を設定し、超過した場合はスキップして続行するオプションを用意する。

レビュー項目:
- [ ] `--split seen` 時に unseen サンプルが出力されないか
- [ ] 既存の WAV ファイルを上書きするか、スキップするかのポリシーが明確か
- [ ] `soundfile` の出力サンプリングレートが 24000 Hz に固定されているか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **Gradio/Streamlit UI**: CLI と並行して Web UI からワンクリック推論できる GUI を最初から用意すると、非エンジニアでも評価に参加できる。
- **バッチ推論最適化（vLLM 的）**: 逐次推論ではなく、複数サンプルを動的バッチングで処理することで推論速度を 3〜5x 向上できる。
- **推論サーバ化**: CLI ではなく FastAPI サーバとして起動し、`/infer` エンドポイントで受け付ける設計にすると、評価パイプラインとの連携が容易になる。

## 7. 後続タスクへの連絡事項

- M4-05〜M4-09 の各評価スクリプトは `output_dir/` の WAV ファイルを入力とする。出力ファイル名の規則（`{sample_id}.wav`）を各チケット担当者に共有すること。
- M4-10 の ablation では同じ CLI を 6 条件の異なるチェックポイントで繰り返す。`--output_dir` を条件ごとに変えることで対応できる設計にすること。
- `testset_seen.json` / `testset_unseen.json` のフォーマット仕様を M4-05〜M4-09 担当者に文書化して渡すこと。
- **M4-12**: バッチ推論完了後（seen/unseen 各 50 発話の全 WAV 生成後）に、生成音声ディレクトリパス（`outputs/seen_results/`, `outputs/unseen_results/`）を M4-12 担当者に共有すること。M4-12 の主観評価はこのディレクトリを参照するため、ファイル数と命名規則を事前に確認すること。
