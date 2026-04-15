# M4-12: 主観評価実施手順書（MOS-Q / MOS-N / SMOS）

> **マイルストーン**: [M4: 推論・評価](../../13_milestones.md#m4-推論評価システム)
> **対応RQ**: RQ-09
> **依存チケット**: M4-04
> **ブロックするチケット**: なし
> **状態**: TODO

---

## 1. 目的とゴール

20 名の評価者による 5 段階主観評価（MOS-Q: 音質、MOS-N: 自然性、SMOS: 参照音声とのメロディ類似度）の実施手順を文書化する。評価フォームの設計、評価者招集方法、信頼区間計算スクリプトを含む。このチケット完了時点で主観評価を実施できる状態になる。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/eval/subjective_ci.py
import numpy as np
from scipy import stats

def compute_mos_ci(scores: list[float], confidence=0.95) -> dict:
    """MOS スコアの平均と 95% 信頼区間を計算する。"""
    n = len(scores)
    mean = np.mean(scores)
    se = stats.sem(scores)  # 標準誤差
    ci = stats.t.interval(confidence, df=n-1, loc=mean, scale=se)
    return {
        "mean": round(float(mean), 3),
        "ci_lower": round(float(ci[0]), 3),
        "ci_upper": round(float(ci[1]), 3),
        "n_raters": n,
    }
```

評価フォームの質問設計:

| 指標 | 質問文 | スケール |
|---|---|---|
| MOS-Q | 「この歌声の音質を評価してください」 | 1(非常に悪い)〜5(非常に良い) |
| MOS-N | 「この歌声の自然さを評価してください」 | 1(非自然)〜5(非常に自然) |
| SMOS | 「参照音声と生成音声のメロディの類似度を評価してください」 | 1(全く異なる)〜5(同一) |

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 設計エージェント | 1 | 評価フォーム設計・`subjective_ci.py` 実装 |
| 運営エージェント | 1 | 評価者招集・データ収集・統計処理 |

## 4. 提供範囲とテスト項目

**含むもの**: `subjective_ci.py` スクリプト、評価フォーム設計書、評価者向けガイドライン

**含まないもの**: 主観評価の実施（人手作業）、自動評価指標（M4-05〜M4-09）

### ユニットテスト

```python
def test_ci_with_known_input():
    scores = [4.0, 4.2, 3.8, 4.1, 3.9] * 4  # 20 名
    result = compute_mos_ci(scores)
    assert 3.5 < result["mean"] < 4.5
    assert result["ci_lower"] < result["mean"] < result["ci_upper"]

def test_single_score_raises():
    with pytest.raises(Exception):
        compute_mos_ci([4.0])  # 自由度 0 でエラー
```

### E2Eテスト

```bash
uv run python models/tts/comelsinger/eval/subjective_ci.py \
  --scores_csv data/subjective_scores.csv \
  --output results/subjective_ci.json
# → {"MOS-Q": {"mean": 4.1, "ci_lower": 3.9, "ci_upper": 4.3, "n_raters": 20}, ...}
```

## 5. 懸念事項とレビュー項目

- **評価者のバイアス**: 20 名全員が歌唱音声に親しんでいる必要はないが、比較評価の基準を統一するために事前キャリブレーション（基準音声の評価）を実施すること。
- **評価疲労**: 1 セッションで評価するサンプル数が多すぎると評価品質が低下する。50 発話 × 3 指標 = 150 回評価を複数セッションに分割すること。

レビュー項目:
- [ ] 評価フォームが Google Forms / Qualtrics で実装可能な設計になっているか
- [ ] 評価者 20 名の招集方法（学内・クラウドソーシング等）が決まっているか
- [ ] 信頼区間が t 分布ベース（小サンプル対応）で計算されているか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **Gradio/Streamlit UI**: 評価フォームを Gradio で実装すると、評価者に URL を共有するだけで開始できる。Google Forms よりも音声再生と評価を同画面で行えて評価品質が向上する。
- **バッチ推論最適化**: 評価サンプルをランダム順に提示するシャッフル機能を最初から組み込み、順序効果を排除する。
- **推論サーバ化**: 評価サーバを常時起動し、評価者がいつでもアクセスできる環境を最初から整備することで、評価収集期間を短縮できる。

## 7. 後続タスクへの連絡事項

- このチケットは M4-04（生成音声の準備）が完了すれば開始できる。ただし評価者 20 名の招集に時間がかかるため、M4-04 完了後すぐに評価者の手配を開始すること。
- 主観評価の結果は M4-13 の最終レポートに組み込む。`subjective_ci.json` のスキーマを M4-13 担当者に事前に共有すること。
- SMOS と SingMOS（M4-07）の相関分析を行い、自動評価の代理妥当性を検証する。両チケットの担当者が連携して分析を実施すること。
