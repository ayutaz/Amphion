# M1-17: バッチ前処理スクリプト (run_preprocess)

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-05
> **依存チケット**: M1-16
> **ブロックするチケット**: M1-18, M2-07
> **状態**: TODO

---

## 1. 目的とゴール

データセットディレクトリを走査し、M1-12〜16 の前処理を全音声ファイルに適用するバッチスクリプトを実装する。

**ゴール**: M4Singer/Opencpop の全音声を前処理し `.pt` ファイルと `metadata.json` を出力。失敗ファイルはスキップして記録。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/run_preprocess.py
import argparse, json, traceback
from pathlib import Path
from tqdm import tqdm

def main():
    args = parse_args()
    models = load_all_models(args)
    manifest = load_manifest(args.data_dir)
    results, errors = [], []

    for item in tqdm(manifest, desc="Preprocessing"):
        try:
            result = process_one(item, args, models)
            results.append(result)
        except Exception as e:
            errors.append({"path": item["wav_path"], "error": str(e)})

    with open(Path(args.output_dir) / "metadata.json", "w") as f:
        json.dump({"results": results, "errors": errors}, f, indent=2)
    print(f"Done: {len(results)} ok, {len(errors)} failed")
```

**実行**: `uv run python models/tts/comelsinger/run_preprocess.py --data_dir /data/m4singer --output_dir /data/preprocessed --device cuda`

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当 |
|---|---|---|
| 実装 | 1 | スクリプト全体 |
| レビュー | 1 | エラーハンドリング・並列化確認 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**:
- `models/tts/comelsinger/run_preprocess.py` スクリプト本体
- `parse_args()` 引数パーサ（`--data_dir`, `--output_dir`, `--device`, `--num_workers` 等）
- `process_one()` ヘルパー関数（M1-12〜16 の前処理を単一ファイルに適用）
- `load_manifest()` ヘルパー（M4Singer/Opencpop の JSON マニフェスト読み込み）
- エラーハンドリング（例外キャッチ → スキップ → errors リストに記録）
- `metadata.json` 出力（`results` と `errors` の2キー構造）

**含まないもの**:
- 各前処理モジュール（M1-12〜16）の実装変更
- データセット固有のマニフェスト変換スクリプト
- 分散処理（マルチノード対応）
- DataLoader 統合（→ M2-07）

### 4.2 ユニットテスト

- `process_one` が例外を投げても `main` がスキップして続行すること
- `metadata.json` が正しいスキーマで出力されること

### 4.3 E2Eテスト

- M1-18 統合テストで合成データ5件に対して `run_preprocess.py` を実行し `metadata.json` が生成されること

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- M4Singer (29時間) を single GPU で処理する時間。`--num_workers` 並列化を検討
- manifest JSON フォーマットの M4Singer/Opencpop データセット構造との整合
- GPU メモリ不足時のチャンク分割（30秒以上の長音声）

### 5.2 レビュー項目

- [ ] エラーログに traceback が含まれているか
- [ ] tqdm の進捗表示が機能しているか
- [ ] `--device cpu` でも動作するか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **Apache Beam/Spark**: 大規模データセット対応の分散処理フレームワーク
- **WebDataset tar 出力**: ストリーミング形式で直接書き出し
- **`--resume` フラグ**: 中断・再実行時に処理済みファイルをスキップ
- **Hydra/OmegaConf**: argparse の代わりに設定管理フレームワーク

## 7. 後続タスクへの連絡事項

- **M2-07**: `metadata.json` の `results` リストが DataLoader の manifest として使用される
- エラー率 1% 超の場合はデータセット品質問題を調査すること
- M1-18 で少数サンプル（5件）に対してスクリプトを実行し動作確認すること
