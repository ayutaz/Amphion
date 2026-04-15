# M1-11: tests/test_losses.py 作成

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-04
> **依存チケット**: M1-07, M1-08, M1-09, M1-10
> **ブロックするチケット**: M2
> **状態**: TODO

---

## 1. 目的とゴール

M1-07〜M1-10 の損失関数全受入基準をカバーする `tests/test_losses.py` を作成する。`torch.manual_seed(42)` による再現性確保と FP16 安定性テストを含む完全なテストスイートを構築する。

**ゴール**: `uv run pytest tests/test_losses.py -v` が全テストパス

## 2. 実装する内容の詳細

```python
import pytest
import torch
import torch.nn.functional as F
from models.tts.comelsinger.losses import (
    compute_scl_loss,
    compute_fcl_loss,
    compute_svt_loss,
    compute_total_loss,
)

@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)

class TestSCLLoss:
    def test_identical_embeddings_loss_zero(self):
        g = torch.randn(8, 512)
        loss = compute_scl_loss(g, g)
        assert loss.item() < 1e-4

    def test_random_loss_positive(self):
        g_a = torch.randn(8, 512)
        g_b = torch.randn(8, 512)
        assert compute_scl_loss(g_a, g_b).item() > 0

class TestFCLLoss:
    def test_all_zero_Y_returns_zero(self):
        B, L, D = 2, 10, 64
        f_a = torch.randn(B, L, D)
        f_b = torch.randn(B, L, D)
        Y = torch.zeros(B, L, L)
        assert compute_fcl_loss(f_a, f_b, Y).item() == 0.0

    def test_fp16_no_nan(self):
        B, L, D = 2, 10, 64
        f_a = torch.randn(B, L, D).half()
        f_b = torch.randn(B, L, D).half()
        Y = torch.eye(L).unsqueeze(0).expand(B, -1, -1).half()
        loss = compute_fcl_loss(f_a, f_b, Y)
        assert not torch.isnan(loss) and not torch.isinf(loss)

class TestSVTLoss:
    def test_all_padding_no_nan(self):
        B, T, V = 2, 20, 129
        logits = torch.randn(B, T, V)
        targets = torch.zeros(B, T, dtype=torch.long)
        durations = torch.ones(B, 5)
        dur_preds = torch.ones(B, 5)
        padding_mask = torch.ones(B, T, dtype=torch.bool)
        total, parts = compute_svt_loss(logits, targets, durations, dur_preds, padding_mask)
        assert not torch.isnan(total)

    def test_dur_pred_equals_target_zero_l_dur(self):
        B, T, V, N = 2, 20, 129, 5
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        dur = torch.rand(B, N) + 0.1
        padding_mask = torch.zeros(B, T, dtype=torch.bool)
        total, parts = compute_svt_loss(logits, targets, dur, dur, padding_mask)
        assert parts["l_dur"].item() < 1e-6

class TestTotalLoss:
    def test_default_weights(self):
        l_scl = torch.tensor(1.0, requires_grad=True)
        l_fcl = torch.tensor(2.0, requires_grad=True)
        l_svt = torch.tensor(3.0, requires_grad=True)
        l_mask = torch.tensor(4.0, requires_grad=True)
        expected = 0.5 * 1.0 + 1.0 * 2.0 + 0.1 * 3.0 + 0.3 * 4.0
        total, parts = compute_total_loss(l_scl, l_fcl, l_svt, l_mask)
        assert abs(total.item() - expected) < 1e-6

    def test_backprop(self):
        l_scl = torch.tensor(1.0, requires_grad=True)
        l_fcl = torch.tensor(1.0, requires_grad=True)
        l_svt = torch.tensor(1.0, requires_grad=True)
        l_mask = torch.tensor(1.0, requires_grad=True)
        total, _ = compute_total_loss(l_scl, l_fcl, l_svt, l_mask)
        total.backward()
        assert l_scl.grad is not None
```

テスト実行:

```bash
uv run pytest tests/test_losses.py -v
uv run pytest tests/test_losses.py --cov=models.tts.comelsinger.losses
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当 |
|---|---|---|
| テスト実装 | 1 | `tests/test_losses.py` 全体の作成・実行確認 |
| テストレビュー | 1 | M1-07〜10 の受入基準との対応確認・境界値補完 |
| CI設定 | 1 | `pyproject.toml` の pytest 設定更新、slow マーク設定 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲

- `tests/test_losses.py`（20+ テスト関数）
- `pyproject.toml` の pytest 設定更新（markers: slow）

### 4.2 ユニットテスト

- SCL: 同一埋め込みで loss ≈ 0、ランダムで loss > 0、tau スケール効果、K_s=1 動作確認
- FCL: Y 全ゼロで loss == 0（exact）、FP16 NaN なし、Y が単位行列で loss ≈ 0
- SVT: 全パディングで NaN なし、dur 一致で l_dur = 0、lambda=0 でバイパス確認
- Total: デフォルト重みの数値一致、バックプロパゲーション成功、dict キー存在確認

### 4.3 E2Eテスト

- `compute_scl_loss → compute_fcl_loss → compute_svt_loss → compute_total_loss` の全パイプラインで `backward()` 成功
- `torch.manual_seed(42)` を設定した同一テストを2回実行して結果が一致すること

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- `compute_svt_loss` の全パディングテストで `l_ce` が NaN になる可能性。実装側の `valid_mask.any()` ガードとの整合性確認が必要
- FP16 テストは CPU では精度が異なる場合があり、CUDA 環境での追加確認が必要
- `test_backprop` でスカラーテンソルを入力とすると実際の S2A forward パスと乖離する

### 5.2 レビュー項目

- [ ] 各テストクラスが対応する M1 チケット番号と受入基準に1:1で対応しているか
- [ ] `autouse=True` の seed fixture がすべてのテストに適用されているか
- [ ] カバレッジ 100% が達成されているか（`--cov` で確認）

## 6. フェーズ振り返り: 一から作り直すとしたら

- **property-based testing**: `hypothesis` で「任意の有効入力で total が有限値かつ非負」を形式検証。手書き境界値テストよりも網羅性が高い
- **数値安定性（BF16）**: FP32 と BF16 の損失値の差が許容範囲内かを `torch.allclose(atol=1e-2)` で検証するテストを最初から追加すべきだった
- **損失関数クラス設計**: 関数テストではなくクラス（`nn.Module`）テストとして設計すれば、パラメータ学習テストも同一ファイルで管理できた

## 7. 後続タスクへの連絡事項

- **M2**: `tests/` の構造を参考に `tests/test_svt_module.py`・`tests/test_s2a_model.py` を作成する
- **CI**: `pyproject.toml` に `[tool.pytest.ini_options] markers = ["slow: marks tests as slow"]` を追加
- 損失関数を変更する場合（lambda 値変更・新成分追加）、本テストスイートの全パスを必須条件とする
