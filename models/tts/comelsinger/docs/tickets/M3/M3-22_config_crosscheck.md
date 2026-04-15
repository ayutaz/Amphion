# M3-22: 設定値クロスチェック

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07
> **依存チケット**: M3-21
> **ブロックするチケット**: M4
> **状態**: TODO

---

## 1. 目的とゴール

要件定義書12の修正値（λ_SCL=1.0, λ_FCL=0.1, λ_SVT=0.5, K_s=8, tau=0.07, scheduler=inverse_sqrt）がコード実装・設定ファイル・ドキュメントの3箇所で一致していることをスクリプトで検証する。M3全体の完了ゲートであり、M4（推論・評価）への移行前に設定の整合性を保証する。

## 2. 実装する内容の詳細

```python
# tools/crosscheck_config.py
import yaml
from models.tts.comelsinger.contrastive_loss import compute_l_cl
import inspect, torch

EXPECTED = {
    "lambda_scl": 1.0,
    "lambda_fcl": 0.1,
    "lambda_cl": 0.5,
    "lambda_svt": 0.5,
    "lambda_mask": 0.3,
    "K_s": 8,
    "tau": 0.07,                    # SCL/FCL の temperature（論文デフォルト値）
    "scheduler": "inverse_sqrt",    # S2A 学習率スケジューラタイプ
}

def crosscheck():
    # 1. 設定ファイル検証
    with open("configs/comelsinger/s2a_train.yaml") as f:
        cfg = yaml.safe_load(f)
    lw = cfg["loss_weights"]
    assert lw["lambda_scl"] == EXPECTED["lambda_scl"]
    assert lw["lambda_fcl"] == EXPECTED["lambda_fcl"]
    assert lw.get("lambda_cl") == EXPECTED["lambda_cl"]
    assert lw["lambda_svt"] == EXPECTED["lambda_svt"]
    assert lw["lambda_mask"] == EXPECTED["lambda_mask"]
    assert cfg["training"]["K_s"] == EXPECTED["K_s"]
    assert cfg["contrastive"].get("tau") == EXPECTED["tau"], \
        f"tau={cfg['contrastive'].get('tau')} != {EXPECTED['tau']}"
    assert cfg["scheduler"]["type"] == EXPECTED["scheduler"], \
        f"scheduler={cfg['scheduler']['type']} != {EXPECTED['scheduler']}"
    print("PASS: config file values OK")

    # 2. コードデフォルト値検証
    sig = inspect.signature(compute_l_cl)
    assert sig.parameters["lambda_scl"].default == EXPECTED["lambda_scl"]
    assert sig.parameters["lambda_fcl"].default == EXPECTED["lambda_fcl"]
    print("PASS: code default values OK")

if __name__ == "__main__":
    crosscheck()
    print("ALL PASS: config crosscheck complete")
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 検証エージェント | 1 | crosscheck_config.py実装・3箇所照合・不一致レポート作成 |

## 4. 提供範囲とテスト項目

**含むもの**: `tools/crosscheck_config.py`、設定ファイル・コード・ドキュメントの3箇所照合

**含まないもの**: 値の修正（不一致が発見された場合は対応チケットで修正）

### ユニットテスト

```bash
uv run python -c "
# EXPECTED定数の値が要件定義書12の修正値と一致することを確認
EXPECTED = {
    'lambda_scl': 1.0, 'lambda_fcl': 0.1, 'lambda_cl': 0.5,
    'lambda_svt': 0.5, 'lambda_mask': 0.3, 'K_s': 8,
    'tau': 0.07, 'scheduler': 'inverse_sqrt',
}
assert EXPECTED['lambda_scl'] == 1.0, 'lambda_SCL should be 1.0 (NOT 0.5 from original paper)'
assert EXPECTED['lambda_fcl'] == 0.1, 'lambda_FCL should be 0.1 (NOT 1.0 from original paper)'
assert EXPECTED['tau'] == 0.07, 'tau (temperature) should be 0.07'
assert EXPECTED['scheduler'] == 'inverse_sqrt', 'scheduler should be inverse_sqrt'
print('PASS: EXPECTED values match requirements doc 12')
"
```

```bash
uv run python -c "
import yaml
with open('configs/comelsinger/s2a_train.yaml') as f:
    cfg = yaml.safe_load(f)
lw = cfg['loss_weights']
mismatches = []
if lw.get('lambda_scl') != 1.0: mismatches.append(f'lambda_scl={lw.get(\"lambda_scl\")} != 1.0')
if lw.get('lambda_fcl') != 0.1: mismatches.append(f'lambda_fcl={lw.get(\"lambda_fcl\")} != 0.1')
if cfg.get('contrastive', {}).get('tau') != 0.07: mismatches.append(f'tau={cfg.get(\"contrastive\", {}).get(\"tau\")} != 0.07')
if cfg.get('scheduler', {}).get('type') != 'inverse_sqrt': mismatches.append(f'scheduler={cfg.get(\"scheduler\", {}).get(\"type\")} != inverse_sqrt')
if mismatches:
    print('FAIL:', mismatches)
else:
    print('PASS: s2a_train.yaml SCL/FCL/tau/scheduler values OK')
"
```

### E2Eテスト

```bash
uv run python tools/crosscheck_config.py
# ALL PASS: config crosscheck complete が出力されること
```

## 5. 懸念事項とレビュー項目

- **論文原著値との乖離**: 論文原著値（λ_SCL=0.5, λ_FCL=1.0）と要件定義書12修正値（λ_SCL=1.0, λ_FCL=0.1）が大きく異なる。クロスチェックレポートにこの差異と選択理由を明記すること。
- **ドキュメント照合の自動化**: CLAUDE.mdやmilestones.mdに記載された値も自動検証の対象に含めることを検討する。

### レビュー項目

- [ ] 設定ファイルの全重み値（lambda_scl, lambda_fcl, lambda_cl, lambda_svt, lambda_mask）が要件定義書12と一致すること
- [ ] `tau=0.07`（SCL/FCL temperature）が設定ファイルの `contrastive.tau` に記載されていること
- [ ] `scheduler.type=inverse_sqrt` が設定ファイルに記載されていること
- [ ] コードのデフォルト値が設定ファイルと一致すること
- [ ] クロスチェックスクリプトがALL PASSで終了すること

## 6. フェーズ振り返り: 一から作り直すとしたら

> 共通の設計判断（PyTorch Lightning vs Accelerate、実験管理、スケジューラ選択）は [M3_design_decisions.md](M3_design_decisions.md) を参照のこと。以下はこのチケット固有の設計判断を記載する。


- **PyTorch Lightning vs 素のAccelerate**: フレームワーク非依存のチェックスクリプト。ただしLightningのHyperParameterログ機能を使えばMLflowやW&Bで設定値が自動記録され、事後クロスチェックが不要になる。
- **実験管理(W&B/MLflow)**: 設定値をW&BのConfigとして記録することで、全実験の設定値をダッシュボードで一覧確認できる。手動クロスチェックスクリプトの代わりになる。
- **学習率スケジューラ選択**: スケジューラのhyperparameter（ウォームアップステップ数、最小lr等）もクロスチェック対象に含めるべきだった。学習率が意図しない値に設定されても気づきにくい。

## 7. 後続タスクへの連絡事項

- **M4**: M3-22のALL PASSをM3フェーズ完了の条件とする。クロスチェックのログをM4開始前のエビデンスとして保存すること。
- **全チケット共通**: 不一致が発見された場合は対応チケット（M3-01: 設定ファイル、M3-13〜M3-17: コード）に差し戻して修正する。
- **将来の参照**: 本クロスチェックスクリプトはM4以降の評価スクリプト開発時にも流用できる。設定値の整合性確認を定期的に実行するCIパイプラインへの組み込みを推奨する。
