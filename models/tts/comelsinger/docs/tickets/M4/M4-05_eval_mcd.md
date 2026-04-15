# M4-05: MCD（Mel Cepstral Distortion）評価スクリプト

> **マイルストーン**: [M4: 推論・評価](../../13_milestones.md#m4-推論評価システム)
> **対応RQ**: RQ-09
> **依存チケット**: M4-04
> **ブロックするチケット**: M4-11, M4-13
> **状態**: TODO

---

## 1. 目的とゴール

生成音声とグラウンドトゥルース音声の MCD（Mel Cepstral Distortion）を計算するスクリプトを実装する。librosa で MFCC を抽出し、DTW で時間長を合わせた上で MCD を算出する。論文の目標値は **4.17 dB**（Seen）。スクリプトは `eval_mcd.py` として単体実行可能であり、JSON 形式でスコアを出力する。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/eval/eval_mcd.py
import librosa, numpy as np
from dtw import dtw   # dtw-python

def compute_mcd(ref_wav: np.ndarray, syn_wav: np.ndarray, sr=24000) -> float:
    """MCD (dB) を DTW アライメント後に計算する。"""
    ref_mfcc = librosa.feature.mfcc(y=ref_wav, sr=sr, n_mfcc=13)[1:]  # 0次除外
    syn_mfcc = librosa.feature.mfcc(y=syn_wav, sr=sr, n_mfcc=13)[1:]
    # DTW アライメント
    alignment = dtw(ref_mfcc.T, syn_mfcc.T, keep_internals=True)
    ref_aligned = ref_mfcc[:, alignment.index1]
    syn_aligned = syn_mfcc[:, alignment.index2]
    diff = ref_aligned - syn_aligned
    mcd = (10.0 / np.log(10.0)) * np.sqrt(2) * np.mean(np.sqrt(np.sum(diff**2, axis=0)))
    return float(mcd)
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `compute_mcd()` 関数・CLI ラッパー実装 |
| 検証エージェント | 1 | 既知の MCD 値との数値照合 |

## 4. 提供範囲とテスト項目

**含むもの**: `compute_mcd()` 関数、50 発話一括評価 CLI、JSON 出力

**含まないもの**: 他の評価指標（F0, MOS など）、ablation 集計（M4-11）

### ユニットテスト

```python
def test_identical_wavs_give_zero_mcd():
    wav = np.random.randn(24000).astype(np.float32)
    mcd = compute_mcd(wav, wav)
    assert mcd < 0.01  # 完全一致 ≈ 0

def test_mcd_is_positive():
    ref = np.random.randn(24000).astype(np.float32)
    syn = np.random.randn(24000).astype(np.float32)
    assert compute_mcd(ref, syn) > 0
```

### E2Eテスト

```bash
uv run python models/tts/comelsinger/eval/eval_mcd.py \
  --ref_dir data/gt_seen/ \
  --syn_dir outputs/seen_results/ \
  --output results/mcd_seen.json
# → {"mean_mcd": 4.17, "std": 0.3, "n_samples": 50}
```

## 5. 懸念事項とレビュー項目

- **DTW のスケール**: `dtw-python` と他ライブラリで距離正規化の方法が異なる。論文の実装と一致するよう確認すること。
- **0次 MFCC の除外**: 多くの実装では 0次成分（エネルギー）を除外する。`n_mfcc=13` で 0 番目（index 0）を除いた 12 次元を使用していることを確認すること。

レビュー項目:
- [ ] `librosa.feature.mfcc` の `n_mfcc` と切り捨て後の次元数が一致しているか
- [ ] 無音区間（GT が短い場合）のパディング処理があるか
- [ ] 出力 JSON のスキーマが M4-13 の期待する形式と一致しているか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **評価パイプライン自動化（CI 統合）**: 評価スクリプトを GitHub Actions に組み込み、PR ごとに MCD を自動計算して差分をコメントする仕組みを最初から設計する。
- **主観評価プラットフォーム選定**: MCD だけでなく、MOS との相関を検証するフローを評価設計の初期段階で組み込む。
- **Ablation 自動実行**: 6 条件の ablation を手動で実行するのではなく、`run_ablation.sh` で一括実行できるスクリプトを最初から用意する。

## 7. 後続タスクへの連絡事項

- M4-11 はこのスクリプトの JSON 出力（`mcd_seen.json`, `mcd_unseen.json`）を読み込んで ablation 集計表を作成する。出力スキーマを M4-11 担当者に事前に共有すること。
- M4-13 は MCD の数値を論文 Table II 形式に整形する。`mean_mcd` と `std` の両方を JSON に含めること。
- 論文の 4.17 dB という目標値は M4Singer の Seen セットに対するもの。Unseen セットの目標値も論文から確認して記録しておくこと。
