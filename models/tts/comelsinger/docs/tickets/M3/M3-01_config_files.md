# M3-01: 学習設定ファイル作成

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07
> **依存チケット**: M2
> **ブロックするチケット**: M3-03
> **状態**: TODO

---

## 1. 目的とゴール

SVT学習用 `svt_train.yaml`（lr=1e-5、50Kステップ、batch=32、fp32）とS2Aファインチューニング用 `s2a_train.yaml`（lr=1e-5、100epoch、batch=32、bf16、K_s=8）の2ファイルを `configs/` 以下に配置する。設定値は要件定義書12の修正値（λ_SCL=1.0, λ_FCL=0.1, λ_SVT=0.5, λ_mask=0.3, K_s=8）を反映する。M3-22でのクロスチェック基準となるため、値の正確性が最重要である。

## 2. 実装する内容の詳細

```yaml
# configs/comelsinger/svt_train.yaml
training:
  max_steps: 50000
  batch_size: 32
  precision: fp32
  seed: 42
optimizer:
  type: AdamW
  lr: 1.0e-5
  weight_decay: 0.01
scheduler:
  type: CosineAnnealingLR
  T_max: 50000
  eta_min: 1.0e-7
```

```yaml
# configs/comelsinger/s2a_train.yaml
training:
  max_epochs: 100
  batch_size: 32
  precision: bf16
  seed: 42
  K_s: 8
loss_weights:
  lambda_scl: 1.0
  lambda_fcl: 0.1
  lambda_cl: 0.5
  lambda_svt: 0.5
  lambda_mask: 0.3
  lambda_seg: 3.0
  lambda_dur: 5.0
lora:
  r: 16
  alpha: 32
  target_modules: [q_proj, v_proj]
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 設定ファイル担当 | 1 | svt_train.yaml と s2a_train.yaml の作成・値検証 |

## 4. 提供範囲とテスト項目

**含むもの**: `configs/comelsinger/svt_train.yaml`, `configs/comelsinger/s2a_train.yaml`

**含まないもの**: OmegaConf/Hydraのスキーマ定義、設定読込コード（→ M3-03）

### ユニットテスト

```bash
uv run python -c "
import yaml
with open('configs/comelsinger/svt_train.yaml') as f:
    cfg = yaml.safe_load(f)
assert cfg['training']['max_steps'] == 50000
assert cfg['training']['batch_size'] == 32
assert cfg['optimizer']['lr'] == 1e-5
print('PASS: svt_train.yaml values OK')
"
```

```bash
uv run python -c "
import yaml
with open('configs/comelsinger/s2a_train.yaml') as f:
    cfg = yaml.safe_load(f)
assert cfg['training']['K_s'] == 8
assert cfg['loss_weights']['lambda_scl'] == 1.0
assert cfg['loss_weights']['lambda_fcl'] == 0.1
assert cfg['loss_weights']['lambda_cl'] == 0.5
print('PASS: s2a_train.yaml values OK')
"
```

### E2Eテスト

```bash
uv run python -c "
import yaml
with open('configs/comelsinger/s2a_train.yaml') as f:
    cfg = yaml.safe_load(f)
lw = cfg['loss_weights']
assert lw['lambda_svt'] == 0.5 and lw['lambda_mask'] == 0.3
assert cfg['lora']['r'] == 16 and cfg['lora']['alpha'] == 32
print('PASS: s2a loss weights and LoRA config OK')
"
```

## 5. 懸念事項とレビュー項目

- **修正値の反映漏れ**: 要件定義書12の修正値（特にλ_SCL=1.0, λ_FCL=0.1）が正確に転記されているか要確認。
- **精度フォーマット**: bf16/fp32の文字列がAccelerateで認識される形式か確認する。

### レビュー項目

- [ ] 要件定義書12との全数値一致確認
- [ ] YAMLパースエラーがないこと
- [ ] K_s=8がs2a_train.yamlに明示されていること

## 6. フェーズ振り返り: 一から作り直すとしたら

- **Hydraの採用**: OmegaConf + Hydraによる階層的設定管理を最初から採用すれば、CLI上書きが容易になり実験管理が改善される。
- **設定スキーマ検証**: pydanticやdataclassesで設定クラスを定義し、型チェックを実施する構成にすべきだった。
- **バージョニング**: 設定ファイルに `version: "1.0"` フィールドを追加し、後方互換性を管理できる構造にする。

## 7. 後続タスクへの連絡事項

- **M3-03**: 設定読込には `yaml.safe_load()` を使用し、辞書アクセスで各値を取得する実装を前提とする。
- **M3-08**: s2a_train.yaml の `lora` セクションをそのままPEFT `LoraConfig` に渡す実装を想定している。
- **M3-22**: 本ファイルの値がコード実装と一致するかの最終クロスチェックを行う基準ファイルとなる。
