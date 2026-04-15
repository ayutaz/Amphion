# M3-12: Algorithm 1 S2A forward（ピッチ条件付け）

> **マイルストーン**: [M3: 学習パイプライン](../../13_milestones.md#m3-学習パイプライン構築)
> **対応RQ**: RQ-07b
> **依存チケット**: M3-11
> **ブロックするチケット**: M3-13, M3-14, M3-16, M3-17
> **状態**: TODO

---

## 1. 目的とゴール

`cond_B = cond_emb + pitch_emb(original)` および `cond_Bp = cond_emb + pitch_emb(perturbed)` の2種類の条件付き埋め込みを計算する処理を実装する。`cond_emb` はプロンプト音声から抽出した音色埋め込み、`pitch_emb` はCoMelSinger_S2Aのピッチ埋め込み層からの出力。この2つの条件付け埋め込みが後続のSCL/FCL/L_mask計算の入力となる。

## 2. 実装する内容の詳細

```python
# tools/train_s2a.py の学習ループ内（Algorithm 1実装部）

def compute_conditioned_embeddings(model, batch, pitch_tokens_orig, pitch_tokens_pert):
    """プロンプト音色埋め込みとピッチ埋め込みを合算して条件付け埋め込みを作成する。

    Returns:
        cond_B: (B, T, D) original ピッチ条件付き埋め込み
        cond_Bp: (B, T, D) perturbed ピッチ条件付き埋め込み
        hidden_states: (B, T, D) S2A backbone の中間表現（SCL/FCL用）
    """
    prompt_audio = batch["prompt_acoustic_tokens"]  # (B, L, 8)
    cond_emb = model.encode_prompt(prompt_audio)    # (B, T, D) - 音色のみ

    pitch_emb_orig = model.pitch_embed(pitch_tokens_orig)   # (B, T, D)
    pitch_emb_pert = model.pitch_embed(pitch_tokens_pert)   # (B, T, D)

    cond_B  = cond_emb + pitch_emb_orig   # original
    cond_Bp = cond_emb + pitch_emb_pert   # perturbed

    hidden_states = model.backbone_forward(cond_B)  # SCL/FCL用の中間表現
    return cond_B, cond_Bp, hidden_states
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | cond_B/cond_Bp計算・pitch_embed統合・中間表現の取得 |

## 4. 提供範囲とテスト項目

**含むもの**: `cond_B`, `cond_Bp` の計算ロジック、`compute_conditioned_embeddings()` 関数

**含まないもの**: SCL計算（→ M3-13）、FCL計算（→ M3-14）、L_mask計算（→ M3-16）、SVT損失（→ M3-17）

### ユニットテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A

model = CoMelSinger_S2A()
# pitch_embed属性が存在することを確認
assert hasattr(model, 'pitch_embed'), 'pitch_embed layer not found'
print('PASS: pitch_embed layer exists')
"
```

```bash
uv run python -c "
import torch

D = 1024
cond_emb = torch.randn(4, 100, D)
pitch_emb_orig = torch.randn(4, 100, D)
pitch_emb_pert = torch.randn(4, 100, D)
cond_B = cond_emb + pitch_emb_orig
cond_Bp = cond_emb + pitch_emb_pert
assert cond_B.shape == cond_Bp.shape == (4, 100, D)
assert not torch.allclose(cond_B, cond_Bp)  # originalとperturbedは異なる
print('PASS: cond_B and cond_Bp computed correctly')
"
```

### E2Eテスト

```bash
uv run python -c "
import torch
from models.tts.comelsinger.comelsinger_s2a import CoMelSinger_S2A
from tools.train_s2a import compute_conditioned_embeddings

model = CoMelSinger_S2A()
batch = {
    'prompt_acoustic_tokens': torch.zeros(4, 150, 8, dtype=torch.long),
    'acoustic_tokens': torch.zeros(4, 100, 8, dtype=torch.long),
}
pitch_orig = torch.zeros(4, 100, dtype=torch.long)
pitch_pert = torch.zeros(4, 100, dtype=torch.long)
cond_B, cond_Bp, hidden = compute_conditioned_embeddings(model, batch, pitch_orig, pitch_pert)
assert cond_B.shape[0] == 4
print('PASS: compute_conditioned_embeddings E2E OK')
"
```

## 5. 懸念事項とレビュー項目

- **encode_prompt の定義**: CoMelSinger_S2A の `encode_prompt()` メソッドが M2 で実装されているか確認する。存在しない場合は本チケット内で追加する。
- **次元の一致**: `cond_emb` と `pitch_emb` の次元 D が一致していること。D=1024（MaskGCT_S2Aの hidden_size）を前提とする。

### レビュー項目

- [ ] `cond_B` と `cond_Bp` のshapeが同一であること
- [ ] `cond_B != cond_Bp`（originalとperturbedが異なること）
- [ ] `pitch_emb` が CoMelSinger_S2A の `pitch_embed` 層から取得されていること

## 6. フェーズ振り返り: 一から作り直すとしたら

> 共通の設計判断（PyTorch Lightning vs Accelerate、実験管理、スケジューラ選択）は [M3_design_decisions.md](M3_design_decisions.md) を参照のこと。以下はこのチケット固有の設計判断を記載する。


- **PyTorch Lightning vs 素のAccelerate**: LightningのforwardメソッドにAlgorithm 1を統合する設計にすれば、conditioned embeddingの計算をモデル内部に閉じ込められる。
- **実験管理(W&B/MLflow)**: `cond_B` と `cond_Bp` のコサイン類似度をW&Bに記録することで、対照学習の進行状況を定量的にモニタリングできる。
- **GradNorm動的重み調整**: `pitch_emb` の貢献度を `cond_emb` と比較しながら動的に調整するGradNorm的アプローチを検討すべきだった。

## 7. 後続タスクへの連絡事項

- **M3-13**: SCLには `cond_B` と `cond_Bp` からAvgPoolで得たグローバル表現を使用する。`cond_B.mean(dim=1)` で取得できることを前提とする。
- **M3-14**: FCLには `hidden_states[K_s:]`（FCL用サンプルの中間表現）を使用する。`hidden_states` の取得方法をM3-12で確定させること。
- **M3-16**: L_maskの計算では `cond_B`（original条件付け）を使用する。perturbedは対照学習専用であり、mask lossには使わない。
- **M3-17**: frozen SVTへの入力も `cond_B` を使用するか `acoustic_tokens` を直接使用するかを設計段階で確認すること。
