# M4-06: F0-RMSE 評価スクリプト

> **マイルストーン**: [M4: 推論・評価](../../13_milestones.md#m4-推論評価システム)
> **対応RQ**: RQ-09
> **依存チケット**: M4-04
> **ブロックするチケット**: M4-11, M4-13
> **状態**: TODO

---

## 1. 目的とゴール

生成音声のピッチ精度を F0-RMSE（半音スケール）で定量評価するスクリプトを実装する。pyworld で F0 を抽出し、無声区間（F0=0）を除外した上で RMSE を算出する。論文の目標値は **0.042**（Seen セミトーン RMSE）。スクリプトは `eval_f0_rmse.py` として独立実行可能であり、JSON 形式でスコアを出力する。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/eval/eval_f0_rmse.py
import pyworld as pw, numpy as np

def hz_to_semitone(f0: np.ndarray) -> np.ndarray:
    """Hz → セミトーン（A4=440Hz 基準）。無声(0)はそのまま 0 を返す。"""
    voiced = f0 > 0
    semitone = np.zeros_like(f0)
    semitone[voiced] = 12.0 * np.log2(f0[voiced] / 440.0) + 69.0
    return semitone

def compute_f0_rmse(ref_wav: np.ndarray, syn_wav: np.ndarray, sr=24000) -> float:
    ref_f0, _ = pw.harvest(ref_wav.astype(np.float64), sr)
    syn_f0, _ = pw.harvest(syn_wav.astype(np.float64), sr)
    # voiced フレームのみ（両方が有声の箇所）で評価
    voiced = (ref_f0 > 0) & (syn_f0 > 0)
    if voiced.sum() == 0:
        return float("nan")
    ref_st = hz_to_semitone(ref_f0)[voiced]
    syn_st = hz_to_semitone(syn_f0)[voiced]
    return float(np.sqrt(np.mean((ref_st - syn_st) ** 2)))
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | `compute_f0_rmse()` 関数・CLI ラッパー実装 |
| 検証エージェント | 1 | pyworld 出力と crepe 出力の比較検証 |

## 4. 提供範囲とテスト項目

**含むもの**: `compute_f0_rmse()` 関数、50 発話一括評価 CLI、JSON 出力

**含まないもの**: F0 抽出アルゴリズムの選択（pyworld 固定）、crepe 代替実装

### ユニットテスト

```python
def test_identical_wavs_give_zero_rmse():
    wav = np.sin(2 * np.pi * 440 * np.arange(24000) / 24000).astype(np.float32)
    rmse = compute_f0_rmse(wav, wav)
    assert rmse < 0.1  # 完全一致 ≈ 0

def test_voiced_filter_excludes_silence():
    ref = np.zeros(24000, dtype=np.float32)  # 無音
    syn = np.random.randn(24000).astype(np.float32)
    rmse = compute_f0_rmse(ref, syn)
    assert np.isnan(rmse)  # voiced フレームなし
```

### E2Eテスト

```bash
uv run python models/tts/comelsinger/eval/eval_f0_rmse.py \
  --ref_dir data/gt_seen/ \
  --syn_dir outputs/seen_results/ \
  --output results/f0_rmse_seen.json
# → {"mean_f0_rmse": 0.042, "std": 0.01, "n_voiced_frames": 12345}
```

## 5. 懸念事項とレビュー項目

- **F0 抽出アルゴリズムの選択**: pyworld は高速だが、crepe（深層学習ベース）より精度が劣る場合がある。論文が使用したアルゴリズムを確認すること（Section IV-C）。
- **DTW による時間アライメント**: GT と生成音声の長さが異なる場合、DTW でアライメントしてから RMSE を計算する必要があるか確認すること。

レビュー項目:
- [ ] セミトーン変換の式（`12 * log2(f0/440) + 69`）が正しいか
- [ ] 両方が無声のフレームを除外しているか（片方のみ無声は含めるか）
- [ ] 出力 JSON のスキーマが M4-13 の期待する形式と一致しているか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **評価パイプライン自動化（CI 統合）**: F0-RMSE を PR ごとに自動計算し、前回コミットとの差分をグラフで可視化する CI ジョブを最初から設計する。
- **主観評価プラットフォーム選定**: F0-RMSE と SMOS（主観メロディ類似度）の相関を分析するフローを評価設計段階で組み込む。
- **Ablation 自動実行**: 6 条件の F0-RMSE を一括計算するラッパースクリプトを最初から用意する。

## 7. 後続タスクへの連絡事項

- M4-11 はこの JSON 出力を読み込んで ablation 集計表を作成する。`mean_f0_rmse` と `std` を必ず JSON に含めること。
- M4-13 は F0-RMSE を論文 Table III 形式に整形する。Seen/Unseen 両セットの結果が必要なため、両方の JSON を生成すること。
- pyworld のインストールに失敗する場合、`uv add pyworld` で解決できる。macOS では `brew install portaudio` が前提になる場合がある。
