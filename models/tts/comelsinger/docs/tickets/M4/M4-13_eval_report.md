# M4-13: 評価レポート生成スクリプト（論文 Table II/III 形式）

> **マイルストーン**: [M4: 推論・評価](../../13_milestones.md#m4-推論評価システム)
> **対応RQ**: RQ-09
> **依存チケット**: M4-05, M4-06, M4-07, M4-08, M4-09, M4-11
> **ブロックするチケット**: なし
> **状態**: TODO

---

## 1. 目的とゴール

全自動評価指標（MCD, F0-RMSE, SingMOS, SECS, SVT-F1）の結果を JSON と Markdown 表で出力するスクリプトを実装する。論文 Table II（客観評価）および Table III（メロディ精度比較）の形式に対応する。このチケット完了時点で、論文の数値との比較が視覚的に確認できる状態になる。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/eval/generate_report.py
import json, pandas as pd
from pathlib import Path

METRIC_FILES = {
    "MCD":      ("mcd_seen.json", "mcd_unseen.json"),
    "F0-RMSE":  ("f0_rmse_seen.json", "f0_rmse_unseen.json"),
    "SingMOS":  ("singmos_seen.json", "singmos_unseen.json"),
    "SECS":     ("secs_seen.json", "secs_unseen.json"),
    "SVT-F1":   ("svt_f1_seen.json", "svt_f1_unseen.json"),
}
PAPER_TARGETS = {  # 論文報告値（参考値として表に併記）
    "MCD":     {"seen": 4.17},
    "F0-RMSE": {"seen": 0.042},
    "SingMOS": {"seen": 4.32},
    "SECS":    {"seen": 0.912},
    "SVT-F1":  {"seen": 0.711},
}

def build_report(results_dir: str) -> dict:
    report = {}
    for metric, (seen_file, unseen_file) in METRIC_FILES.items():
        seen = json.load(open(Path(results_dir) / seen_file))
        unseen = json.load(open(Path(results_dir) / unseen_file))
        key = f"mean_{metric.lower().replace('-','_')}"
        report[metric] = {
            "seen":   seen.get(key),
            "unseen": unseen.get(key),
            "paper_target_seen": PAPER_TARGETS[metric].get("seen"),
        }
    return report
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `generate_report.py` 本体・JSON/MD 出力 |
| 検証エージェント | 1 | 論文 Table II/III との数値照合 |

## 4. 提供範囲とテスト項目

**含むもの**: `generate_report.py` スクリプト、JSON + Markdown 出力、論文目標値との差分表示

**含まないもの**: ablation 集計（M4-11）、主観評価（M4-12）の統合（別テーブル）

### ユニットテスト

```python
def test_report_contains_all_metrics(results_dir_fixture):
    report = build_report(results_dir_fixture)
    assert set(report.keys()) == {"MCD", "F0-RMSE", "SingMOS", "SECS", "SVT-F1"}

def test_report_includes_paper_targets(results_dir_fixture):
    report = build_report(results_dir_fixture)
    assert report["MCD"]["paper_target_seen"] == 4.17
```

### E2Eテスト

```bash
uv run python models/tts/comelsinger/eval/generate_report.py \
  --results_dir results/ \
  --output results/eval_report.json
cat results/eval_report.md
# → Table II形式：| Metric | Seen | Unseen | Paper(Seen) |
```

## 5. 懸念事項とレビュー項目

- **論文の比較モデル**: Table II には CoMelSinger だけでなく比較手法（MaskGCT, etc.）の数値も記載されている。本スクリプトは CoMelSinger の数値のみを扱い、比較手法は手動で追記する設計でよいか確認すること。
- **JSON キー名の統一**: M4-05〜M4-09 の各 JSON で使用しているキー名（`mean_mcd`, `mean_f0_rmse` など）がこのスクリプトと一致していることを事前に確認すること。

レビュー項目:
- [ ] Markdown 表が論文 Table II の列順と一致しているか
- [ ] JSON に `paper_target_seen` が含まれており、実装値との差分（`delta`）が計算されているか
- [ ] Seen/Unseen 両セットの数値が揃っているか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **評価パイプライン自動化（CI 統合）**: `generate_report.py` を CI の最終ステップとして自動実行し、PR に Table II/III の差分をコメントとして投稿する仕組みを最初から設計する。
- **主観評価プラットフォーム選定**: 主観評価（M4-12）の結果もこのレポートに統合できるよう、JSON スキーマを最初から統一設計する。
- **Ablation 自動実行**: M4-11 の ablation 表と M4-13 の最終レポートを同一スクリプトで生成できるよう、最初から統合した設計にする。

## 7. 後続タスクへの連絡事項

- このチケットは M4-05〜M4-09 の全評価 JSON が揃っていることを前提とする。いずれかが欠けている場合は `null` で埋めて表を生成するか、エラーを出すかを事前に決めておくこと。
- M4-11 の ablation 表（`ablation_table_v.md`）はこのレポートとは別ファイルとして出力する。最終成果物として両方を `results/` ディレクトリに格納すること。
- 論文投稿前に Table II/III の数値を最終確認するレビューを実施すること。このチケットの出力が最終確認の起点となる。
