# M4-10: Ablation Study 設定 YAML 6 条件

> **マイルストーン**: [M4: 推論・評価](../../13_milestones.md#m4-推論評価システム)
> **対応RQ**: RQ-09
> **依存チケット**: M4-01
> **ブロックするチケット**: M4-11
> **状態**: TODO

---

## 1. 目的とゴール

論文 Table V の ablation study を再現するための 6 条件設定ファイルを YAML で作成する。条件は `full / wo_cl / wo_scl / wo_fcl / wo_svt / wo_cl_svt` の 6 種類。各 YAML はベース設定を継承し、差分のみ上書きする構造にする。これにより M4-04 の `run_inference.py` が `--config` 引数で各条件を切り替えられるようになる。

## 2. 実装する内容の詳細

```yaml
# configs/ablation/wo_cl.yaml — 対照学習なし
_base_: ../comelsinger_base.yaml

model:
  s2a:
    loss:
      lambda_scl: 0.0   # SCL 無効化
      lambda_fcl: 0.0   # FCL 無効化
    # LoRA チェックポイントは wo_cl 専用のものを指定
    lora_ckpt_dir: checkpoints/ablation/wo_cl/lora
    pitch_emb_path: checkpoints/ablation/wo_cl/pitch_emb.pt
```

6 条件一覧:

| 条件 ID | ファイル名 | 変更箇所 |
|---|---|---|
| full | full.yaml | 変更なし（全コンポーネント有効） |
| wo_cl | wo_cl.yaml | λ_SCL=0, λ_FCL=0 |
| wo_scl | wo_scl.yaml | λ_SCL=0 のみ |
| wo_fcl | wo_fcl.yaml | λ_FCL=0 のみ |
| wo_svt | wo_svt.yaml | λ_SVT=0 |
| wo_cl_svt | wo_cl_svt.yaml | λ_SCL=0, λ_FCL=0, λ_SVT=0 |

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 設定エージェント | 1 | 6 つの YAML ファイル作成・継承構造設計 |
| 検証エージェント | 1 | 各 YAML を読み込んで設定値が正しく上書きされることを確認 |

## 4. 提供範囲とテスト項目

**含むもの**: 6 条件 YAML ファイル、ベース設定継承構造

**含まないもの**: 各条件のモデル学習（M3 の範囲）、評価指標計算（M4-05〜M4-09）

### ユニットテスト

```python
def test_wo_cl_disables_contrastive():
    cfg = OmegaConf.load("configs/ablation/wo_cl.yaml")
    assert cfg.model.s2a.loss.lambda_scl == 0.0
    assert cfg.model.s2a.loss.lambda_fcl == 0.0

def test_full_enables_all():
    cfg = OmegaConf.load("configs/ablation/full.yaml")
    assert cfg.model.s2a.loss.lambda_scl == 1.0
    assert cfg.model.s2a.loss.lambda_fcl == 0.1
    assert cfg.model.s2a.loss.lambda_cl == 0.5
    assert cfg.model.s2a.loss.lambda_svt == 0.5
```

### E2Eテスト

```bash
for condition in full wo_cl wo_scl wo_fcl wo_svt wo_cl_svt; do
  uv run python models/tts/comelsinger/run_inference.py \
    --config configs/ablation/${condition}.yaml \
    --testset data/testset_seen.json \
    --output_dir outputs/ablation/${condition}
done
ls outputs/ablation/  # → 6 ディレクトリ
```

## 5. 懸念事項とレビュー項目

- **継承の深さ**: `_base_` による多段継承はデバッグが難しくなる。最大 2 段（base → ablation）に制限すること。
- **チェックポイントパスの管理**: 各条件で異なるチェックポイントを参照するため、存在しないパスを指定してもエラーが推論開始まで出ない。起動時にパス存在確認をすること。

レビュー項目:
- [ ] `full.yaml` が要件定義書12確定値（λ_SCL=1.0, λ_FCL=0.1, λ_CL=0.5, λ_SVT=0.5）を正確に反映しているか
- [ ] 各 YAML のチェックポイントパスが実際のディレクトリ構造と一致しているか
- [ ] `OmegaConf.merge` での上書き順序が正しいか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **評価パイプライン自動化（CI 統合）**: 設定 YAML の変更を検知して自動的に ablation を再実行する CI パイプラインを最初から設計する。
- **主観評価プラットフォーム選定**: ablation 条件の違いが主観評価（MOS-Q/N）にどう影響するかを評価できるプラットフォームを選定段階から組み込む。
- **Ablation 自動実行**: 6 条件の学習・推論・評価を一括実行するオーケストレーションスクリプト（Makefile や Taskfile）を最初から用意する。

## 7. 後続タスクへの連絡事項

- M4-11 は 6 条件の評価結果を集計する。各条件の出力ディレクトリ名（`outputs/ablation/{condition}/`）を M4-11 担当者に伝えること。
- M4-04 の `--config` 引数がこの YAML パスを正しく受け取れるよう、パス解決ロジックを M4-04 担当者と確認すること。
- 各条件のチェックポイントが M3 の学習完了後に揃うまで、M4-11 の実行を待機する必要がある。依存関係を M4-11 担当者に伝えること。
