# M4-01: CoMelSinger 推論パイプライン基盤クラス

> **マイルストーン**: [M4: 推論・評価](../../13_milestones.md#m4-推論評価システム)
> **対応RQ**: RQ-08
> **依存チケット**: M3（S2A 学習パイプライン完了）
> **ブロックするチケット**: M4-02
> **状態**: TODO

---

## 1. 目的とゴール

MaskGCT の `MaskGCT_Inference_Pipeline` を継承し、歌唱音声合成に対応した `CoMelSinger_Inference_Pipeline` を実装する。ピッチトークナイザーの保持・初期化を基盤クラスに組み込み、後続チケット（M4-02）が `comelsinger_inference()` を安全に呼び出せる状態にする。このチケット完了時点では pipeline の初期化と基本属性の検証が通ること。

## 2. 実装する内容の詳細

```python
# models/tts/comelsinger/comelsinger_inference.py
from models.tts.maskgct.maskgct_inference import MaskGCT_Inference_Pipeline
from models.tts.comelsinger.pitch_tokenizer import PitchTokenizer

class CoMelSinger_Inference_Pipeline(MaskGCT_Inference_Pipeline):
    def __init__(self, cfg, device="cuda"):
        super().__init__(cfg, device=device)
        self.pitch_tokenizer = PitchTokenizer(
            n_bins=cfg.pitch.n_bins,       # デフォルト 128
            f0_min=cfg.pitch.f0_min,        # デフォルト 50.0 Hz
            f0_max=cfg.pitch.f0_max,        # デフォルト 1100.0 Hz
        )
        self.device = device
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実装エージェント | 1 | クラス定義・継承構造・`__init__` 実装 |
| レビューエージェント | 1 | 親クラスの `__init__` シグネチャ確認・属性衝突検査 |

## 4. 提供範囲とテスト項目

**含むもの**: `CoMelSinger_Inference_Pipeline` クラス定義、`__init__` 実装、`pitch_tokenizer` 属性

**含まないもの**: 実際の推論メソッド（→ M4-02）、モデル重みロード（→ M4-03）

### ユニットテスト

```python
# uv run python -m pytest tests/test_m4_01.py -v
def test_pipeline_init(cfg_fixture):
    pipe = CoMelSinger_Inference_Pipeline(cfg_fixture, device="cpu")
    assert hasattr(pipe, "pitch_tokenizer")

def test_pitch_tokenizer_attr(cfg_fixture):
    pipe = CoMelSinger_Inference_Pipeline(cfg_fixture, device="cpu")
    assert pipe.pitch_tokenizer.n_bins == 128
```

### E2Eテスト

```bash
uv run python -c "
from models.tts.comelsinger.comelsinger_inference import CoMelSinger_Inference_Pipeline
print('import OK')
"
```

## 5. 懸念事項とレビュー項目

- **親クラスの `__init__` 変更**: MaskGCT 側の更新で `super().__init__` シグネチャが変わるリスクがある。バージョンをピン留めして確認すること。
- **`device` の二重管理**: 親クラスも `self.device` を持つ場合、上書きしないよう注意。

レビュー項目:
- [ ] 親クラスの全属性が `CoMelSinger_Inference_Pipeline` で参照可能か
- [ ] `PitchTokenizer` の import パスが正しいか
- [ ] `device="cpu"` で CPU 実行できるか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **Gradio/Streamlit UI を最初から設計**: 推論パイプラインを CLI 専用に作ると、後から UI を追加するときに大幅なリファクタが必要になる。最初から `pipeline.infer()` を UI 非依存な関数として切り出しておくべき。
- **バッチ推論最適化（vLLM 的）**: 逐次推論を前提としたクラス設計は、複数リクエストの同時処理に対応できない。KV キャッシュや動的バッチングを考慮したインターフェース設計を初期から行うべき。
- **推論サーバ化を想定した依存注入**: `pitch_tokenizer` を `__init__` で固定するのではなく、DI コンテナ経由で差し替え可能にすると A/B テストが容易になる。

## 7. 後続タスクへの連絡事項

- M4-02 は `CoMelSinger_Inference_Pipeline` インスタンスを受け取り `comelsinger_inference()` を実装する。このチケットが完了するまで M4-02 は開始不可。
- M4-03 は `from models.tts.comelsinger.comelsinger_inference import CoMelSinger_Inference_Pipeline` を前提とする。
- `cfg` の構造（`cfg.pitch.n_bins` 等）を M4-02 担当者に事前に共有すること。
