# M4-11: Ablation Study 集計スクリプト

> **マイルストーン**: [M4: 推論・評価](../../13_milestones.md#m4-推論評価システム)
> **対応RQ**: RQ-09
> **依存チケット**: M4-05, M4-06, M4-07, M4-08, M4-09, M4-10
> **ブロックするチケット**: M4-13
> **状態**: TODO

---

## 1. 目的とゴール

6 条件 × 5 指標（MCD, F0-RMSE, SingMOS, SECS, SVT-F1）のクロス集計表を生成するスクリプトを実装する。各条件の評価 JSON を読み込み、CSV と Markdown 形式で論文 Table V に対応した集計表を出力する。このチケット完了時点で M4-13 がレポートを組み立てるための全データが揃う。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/eval/ablation_summary.py
import json, pandas as pd
from pathlib import Path

CONDITIONS = ["full", "wo_cl", "wo_scl", "wo_fcl", "wo_svt", "wo_cl_svt"]
METRICS = ["mcd", "f0_rmse", "singmos", "secs", "svt_f1"]

def load_results(results_dir: str) -> pd.DataFrame:
    rows = []
    for cond in CONDITIONS:
        row = {"condition": cond}
        for metric in METRICS:
            json_path = Path(results_dir) / cond / f"{metric}_seen.json"
            data = json.load(open(json_path))
            row[f"{metric}_mean"] = data[f"mean_{metric}"]
            row[f"{metric}_std"] = data.get("std", None)
        rows.append(row)
    return pd.DataFrame(rows)

def save_summary(df: pd.DataFrame, out_dir: str):
    out = Path(out_dir)
    df.to_csv(out / "ablation_table_v.csv", index=False)
    with open(out / "ablation_table_v.md", "w") as f:
        f.write(df.to_markdown(index=False))
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `load_results()` / `save_summary()` 実装・CLI ラッパー |
| 検証エージェント | 1 | 出力 CSV/MD の数値が各 JSON と一致することを確認 |

## 4. 提供範囲とテスト項目

**含むもの**: `ablation_summary.py` スクリプト、CSV/Markdown 出力

**含まないもの**: 評価指標計算（M4-05〜M4-09）、最終レポート整形（M4-13）

### ユニットテスト

```python
def test_all_conditions_present(results_dir_fixture):
    df = load_results(results_dir_fixture)
    assert set(df["condition"]) == set(CONDITIONS)

def test_no_missing_values(results_dir_fixture):
    df = load_results(results_dir_fixture)
    assert not df.isnull().any().any()
```

### E2Eテスト

```bash
uv run python models/tts/comelsinger/eval/ablation_summary.py \
  --results_dir outputs/ablation/ \
  --output_dir results/
cat results/ablation_table_v.md
# → 6行×5指標のMarkdown表
```

## 5. 懸念事項とレビュー項目

- **JSON スキーマの統一**: M4-05〜M4-09 の JSON キー名（`mean_mcd`, `mean_f0_rmse` など）がスクリプト内のキー名と一致していることを事前に確認すること。
- **欠損値の扱い**: 特定条件の評価 JSON が存在しない場合、`NaN` で埋めて表を生成するか、エラーで停止するかを決めておくこと。

レビュー項目:
- [ ] CSV の列名が M4-13 の期待するスキーマと一致しているか
- [ ] Markdown 表が論文 Table V の列順と一致しているか
- [ ] `full` 条件の数値が論文の報告値に近いか（数値的な妥当性チェック）

## 6. フェーズ振り返り: 一から作り直すとしたら

- **評価パイプライン自動化（CI 統合）**: ablation 集計を CI の最終ステップとして自動実行し、PR に Table V の差分をコメントする仕組みを最初から設計する。
- **主観評価プラットフォーム選定**: 自動評価 Table V と主観評価（M4-12）を同一フォーマットで比較できる統合レポートを最初から設計する。
- **Ablation 自動実行**: `ablation_summary.py` を単体で動かすだけでなく、M4-10 の推論からここまでを一括実行するパイプラインスクリプトを最初から用意する。

## 7. 後続タスクへの連絡事項

- M4-13 はこのチケットの出力（`ablation_table_v.csv`, `ablation_table_v.md`）を最終レポートに組み込む。CSV のカラム名を M4-13 担当者に事前に共有すること。
- 6 条件のうちいずれかの評価 JSON が未完成の場合、このチケットは実行できない。M4-05〜M4-09 の完了ステータスを確認してから開始すること。
- Seen/Unseen 両セットの集計表（`ablation_table_v_seen.csv` と `ablation_table_v_unseen.csv`）を分けて出力するか、単一ファイルに統合するかを M4-13 担当者と合意すること。
