# M3-17: Algorithm 1 SVT損失（L_SVT）

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07b
> **依存チケット**: M3-12
> **ブロックするチケット**: M3-18
> **状態**: TODO

---

## 1. 目的とゴール

`torch.no_grad()` コンテキスト内でfrozen SVTモデルを動かし `compute_svt_loss()` を計算する。SVTモジュールには勾配が流れないことを確認する（StopGrad）。L_SVTはS2A学習中に固定ピッチ監督として機能し、生成された音響トークンがターゲットピッチと一致するよう間接的に誘導する。

## 2. 実装する内容の詳細

```python
# tools/train_s2a.py の学習ループ内（L_SVT計算部分）

def compute_l_svt(
    svt_model,
    acoustic_tokens: torch.Tensor,
    pitch_tokens: torch.Tensor,
) -> torch.Tensor:
    """frozen SVTを用いてピッチ監督損失を計算する。

    SVTには勾配を流さない（StopGrad）。
    S2Aが生成するacoustic_tokensがSVTで正しいピッチに変換できるか監督する。

    Args:
        svt_model: frozen SVTModule（requires_grad=False）
        acoustic_tokens: (B, T, Q) 音響トークン
        pitch_tokens: (B, T) 正解ピッチトークン

    Returns:
        l_svt: SVT損失（スカラー）
    """
    with torch.no_grad():
        svt_out = svt_model(acoustic_tokens)
    loss_dict = svt_model.compute_svt_loss_from_logits(
        logits=svt_out["logits"].detach(),
        pitch_tokens=pitch_tokens,
    )
    return loss_dict["loss"]
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | compute_l_svt()実装・no_grad確認・SVT grad検証 |

## 4. 提供範囲とテスト項目

**含むもの**: `compute_l_svt()` 関数、`torch.no_grad()` によるSVT grad遮断

**含まないもの**: SVTの内部損失実装（→ M2）、L_total統合（→ M3-18）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.svt_module import SVTModule

# SVTがfrozenになっていること確認
svt = SVTModule()
for p in svt.parameters():
    p.requires_grad_(False)
assert not any(p.requires_grad for p in svt.parameters())
print('PASS: SVT all params frozen')
"
```

```bash
uv run python -c "
import torch

# torch.no_grad()内でも損失の計算自体は可能なことを確認
x = torch.randn(4, 10, requires_grad=True)
with torch.no_grad():
    y = x * 2.0
    loss = y.mean()
assert not loss.requires_grad
print('PASS: no_grad computation OK, loss.requires_grad=False')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.svt_module import SVTModule
from tools.train_s2a import compute_l_svt

svt = SVTModule()
for p in svt.parameters(): p.requires_grad_(False)
acoustic = torch.zeros(4, 50, 8, dtype=torch.long)
pitch = torch.zeros(4, 50, dtype=torch.long)
l_svt = compute_l_svt(svt, acoustic, pitch)
assert not torch.isnan(l_svt) and l_svt.requires_grad == False
print(f'PASS: L_SVT={l_svt.item():.4f}, requires_grad={l_svt.requires_grad}')
"
```

## 5. 懸念事項とレビュー項目

- **compute_svt_loss_from_logits の設計**: SVTのforwardとloss計算を分離する必要がある。`torch.no_grad()` でlogitsを取得し、その後 `.detach()` でgrpah切断した上でlossを計算する。
- **acoustic_tokens の勾配**: L_SVTはS2Aの生成したacoustic_tokensを入力とするが、学習中はacoustic_tokensがstraight-throughで勾配を受け取るか確認が必要。

### レビュー項目

- [ ] SVTの全パラメータに勾配が流れていないこと
- [ ] `torch.no_grad()` が SVTの forward を包んでいること
- [ ] L_SVT が正の有限値であること

## 6. フェーズ振り返り: 一から作り直すとしたら

- **PyTorch Lightning vs 素のAccelerate**: LightningのforwardにSVT呼び出しを組み込む場合、`self.svt_model.eval()` と `torch.no_grad()` の両立をフックで管理できる。
- **実験管理(W&B/MLflow)**: L_SVTのL_CE, L_seg, L_durの3成分を個別にW&Bに記録することで、SVT監督のどの要素がS2Aに効いているか分析できる。
- **GradNorm動的重み調整**: L_SVTの重み(lambda_svt=0.5)を固定するのではなく、GradNormで動的調整することでS2Aの学習安定性が向上する可能性がある。

## 7. 後続タスクへの連絡事項

- **M3-18**: `L_total = 0.5 * L_CL + 0.5 * L_SVT + 0.3 * L_mask` の `L_SVT` として本関数の出力を使用する。lambda_svt=0.5は設定ファイルから読み込む。
- **M3-19**: L_SVT単独をTensorBoardに記録すること。L_SVT内のL_CE, L_seg, L_durも個別記録を推奨する。
- **M3-22**: lambda_svt=0.5 がコード・設定・ドキュメントで一致するかをM3-22でクロスチェックする。（論文原著値はλ_SVT=0.1と異なることに注意）
