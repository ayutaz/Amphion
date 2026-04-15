# CoMelSinger 再現実装 マイルストーン計画書

> **対象論文**: "CoMelSinger: Discrete Token-Based Zero-Shot Singing Synthesis With Structured Melody Control and Guidance" (Zhao et al., 2026, arXiv:2509.19883v2)
> **作成日**: 2026-04-15
> **前提**: 要件定義書 `12_requirements_specification.md` に基づくマイルストーン分解

---

## 全体概要

```
M0: 環境構築          (1-2日)   ← 現在地
 ↓
M1: 基盤モジュール     (5-7日)   RQ-01, RQ-04, RQ-05
 ↓
M2: コアモジュール     (10-13日)  RQ-02, RQ-06, RQ-03
 ↓
M3: 学習パイプライン   (7-10日)  RQ-07a, RQ-07b
 ↓
M4: 推論・評価        (8-11日)  RQ-08, RQ-09

合計推定工数: 31-43日
```

### マイルストーン間の依存関係

```
M0 ──→ M1 ──→ M2 ──→ M3 ──→ M4
              │              │
              └──────────────┘
              M2のデータ前処理成果物を
              M3の学習で使用
```

---

## M0: 環境構築

**目標**: CoMelSinger再現実装の開発・実験が可能な再現性のあるPython環境を構築し、全依存ライブラリおよび事前学習済みモデルを利用可能な状態にする

**前提条件**:
- git sparse-checkout 設定済み（`models/tts/maskgct`, `models/tts/comelsinger` 等がチェックアウト済み）
- `uv` インストール済み
- NVIDIA GPU（RTX A5000 × 4 または同等）が利用可能
- HuggingFace アカウントおよびアクセストークンを取得済み

**推定工数**: 1-2日

### タスク一覧

| ID | タスク | 詳細 | 成果物 | 受入基準 |
|---|---|---|---|---|
| M0-1 | sparse-checkout に `models/codec` を追加 | `git sparse-checkout add models/codec` を実行。MaskGCTの `maskgct_utils.py` が `CodecEncoder`/`CodecDecoder` をインポートするために必須 | `models/codec/` ディレクトリ | `uv run python -c "from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder"` がエラーなし |
| M0-2 | `uv init` でプロジェクト初期化 | リポジトリルートで `uv init` を実行し、`pyproject.toml` を生成 | `pyproject.toml`, `uv.lock` | `uv run python --version` が正常値を返す |
| M0-3 | PyTorch系ライブラリをインストール | `uv add torch>=2.2.0 torchaudio>=2.2.0` を実行。GPU環境ではCUDA対応ビルドを指定 | `uv.lock` にエントリ追加 | `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` がバージョン2.2+かつTrueを返す |
| M0-4 | ML系ライブラリをインストール | `uv add transformers>=4.40.0 peft>=0.11.0 accelerate>=0.30.0` を実行 | `uv.lock` にエントリ追加 | `uv run python -c "from peft import LoraConfig; from transformers import LlamaConfig"` がエラーなし |
| M0-5 | 音声処理系ライブラリをインストール | `uv add pyworld>=0.3.4 librosa>=0.10.1 pypinyin>=0.50.0 mir_eval>=0.7` を実行 | `uv.lock` にエントリ追加 | `uv run python -c "import pyworld; import librosa; import pypinyin"` がエラーなし |
| M0-6 | Amphion固有依存をインストール | `uv add json5 ruamel.yaml tqdm huggingface_hub` を実行 | `uv.lock` にエントリ追加 | `uv run python -c "import json5"` がエラーなし |
| M0-7 | 事前学習済みモデル: MaskGCT-T2S | `huggingface-cli download amphion/MaskGCT-T2S` でT2Sモデル重みを取得 | T2Sモデルファイル | ファイルサイズが0でないこと |
| M0-8 | 事前学習済みモデル: MaskGCT-S2A | `huggingface-cli download amphion/MaskGCT-S2A` でS2Aモデル重み（1layer + full）を取得。LoRA fine-tuningの初期重み | S2Aモデルファイル | `load_state_dict` が `unexpected keys` なしで完了 |
| M0-9 | 事前学習済みモデル: w2v-bert-2.0 | `huggingface-cli download facebook/w2v-bert-2.0` でセマンティック特徴抽出モデルを取得 | HuggingFaceキャッシュ | `uv run python -c "from transformers import Wav2Vec2BertModel; Wav2Vec2BertModel.from_pretrained('facebook/w2v-bert-2.0')"` が完了 |
| M0-10 | 事前学習済みモデル: Amphion Codec | `amphion/MaskGCT` リポジトリから `CodecEncoder/CodecDecoder` の重みを取得 | コーデック重みファイル | 設定ファイルとの整合性を確認 |
| M0-11 | `wav2vec2bert_stats.pt` の配置確認 | `models/tts/maskgct/ckpt/wav2vec2bert_stats.pt` の存在確認 | 既存ファイル確認 | `uv run python -c "import torch; d = torch.load('models/tts/maskgct/ckpt/wav2vec2bert_stats.pt'); print(d.keys())"` が `['mean', 'var']` を返す |
| M0-12 | MaskGCT既存コードの動作確認 | smoke test: MaskGCTモジュールが正常にインポートできることを確認 | 確認ログ | `uv run python -c "from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A; from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S"` がエラーなし |
| M0-13 | `comelsinger/` パッケージ初期化確認 | `models/tts/comelsinger/__init__.py` が存在しインポート可能であることを確認 | 確認ログ | `uv run python -c "from models.tts.comelsinger import *"` がエラーなし |

### 依存関係

- M0-1 → M0-10, M0-12 の前提（CodecEncoder インポートに必要）
- M0-2 → M0-3〜M0-6 の前提（pyproject.toml 生成後に uv add を実行）
- M0-3 → M0-4〜M0-6, M0-12 の前提（torch が他ライブラリの依存元）
- M0-7〜M0-10 は並列実行可能
- M0-12 は M0-1〜M0-6 完了後に実施
- M0-13 は M0-12 完了後に実施

### 完了判定基準

- [ ] `git sparse-checkout list` に `models/codec` が含まれる
- [ ] `uv run python --version` が正常値を返す
- [ ] `uv run python -c "import torch; assert torch.__version__ >= '2.2.0'"` がエラーなし
- [ ] `uv run python -c "import torch; assert torch.cuda.is_available()"` がエラーなし（GPU環境）
- [ ] `uv run python -c "from peft import LoraConfig; from transformers import Wav2Vec2BertModel; import pyworld; import librosa; print('ALL OK')"` が `ALL OK` を出力
- [ ] MaskGCT-T2S / MaskGCT-S2A / w2v-bert-2.0 / Amphion Codec の各ファイルが存在しサイズ非ゼロ
- [ ] `uv run python -c "from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A; from models.codec.amphion_codec.codec import CodecEncoder; print('OK')"` が `OK` を出力
- [ ] `uv.lock` がコミット可能な状態で存在

### リスクと対策

| リスク | 影響度 | 対策 |
|---|---|---|
| `pyworld` のビルドエラー（Apple Silicon等） | 高 | `uv add pyworld` 失敗時は `crepe` をF0抽出の代替として使用 |
| HuggingFaceモデルのリポジトリ構造が不明確 | 中 | `huggingface-cli repo-files amphion/MaskGCT` で事前確認、`maskgct_inference.py` のロードパターンを参照 |
| PyTorch CUDA バージョンとドライバーの不一致 | 高 | `nvidia-smi` で確認し対応CUDA版を `--index-url` で明示指定 |
| Amphionのインポートパスが PYTHONPATH に含まれない | 中 | `pyproject.toml` でルートをソースパスとして設定、または `PYTHONPATH=.` を設定 |

---

## M1: 基盤モジュール実装

**目標**: CoMelSingerの中核となる3ファイル（`pitch_tokenizer.py` / `losses.py` / `preprocess.py`）を実装し、単体テストを全てパスさせる。後続フェーズが安定して依存できる土台を確立する。

**前提条件**: M0完了

**推定工数**: 5-7日

### タスク一覧

| ID | タスク | 対応RQ | 詳細 | 成果物 | 受入基準 |
|---|---|---|---|---|---|
| M1-1 | `PitchTokenizer` クラス骨格 | RQ-01 | `pitch_tokenizer.py` 新規作成。定数 `VOCAB_SIZE=129`, `ENCODEC_FPS=75.0`, `F0_FPS=200.0`、`__init__` のみ | `pitch_tokenizer.py` 骨格 | `PitchTokenizer()` がインスタンス化できる |
| M1-2 | `quantize_f0()` 実装 | RQ-01 | F0配列 `(T_f0,)` → 各EnCodecフレーム区間の有声F0中央値でピッチ決定。MIDI変換 `round(12*log2(f/440)+69)`, `clamp(1,128)`。無声→token 0 | `quantize_f0()` | `quantize_f0([440.0], 1)→[69]`, `quantize_f0([0.0], 1)→[0]`, 出力値 0-128 のみ |
| M1-3 | `tokenize_score()` 実装 | RQ-01 | MIDI番号列+デュレーション列 → ピッチトークン+フレーム数列。デュレーション正規化、`sum(a_d)==target_len` 保証、各 `a_d[i]>=1` 保証 | `tokenize_score()` | `sum(frame_counts)==target_len` をランダム100ケースで検証、`all(frame_counts>=1)` |
| M1-4 | `compute_soft_label_matrix()` 実装 | RQ-01 | `(L,)` → `(L,L)` ソフトラベル行列。`Y[i,j]=1.0` if 同一ピッチ&両方有声、それ以外0。無声同士は正例としない | `compute_soft_label_matrix()` | 対称性検証、無声同士=0、有声同ピッチ=1.0 |
| M1-5 | `decode_token_to_freq()` 実装 | RQ-01 | token → Hz逆変換。`token==0` → `0.0`、`token==n` → `440*2**((n-69)/12)` | `decode_token_to_freq()` | `decode_token_to_freq(69) ≈ 440.0` |
| M1-6 | PitchTokenizer 単体テスト | RQ-01 | `tests/test_pitch_tokenizer.py` 作成。M1-2〜5の受入基準を全カバー | `tests/test_pitch_tokenizer.py` | `uv run pytest tests/test_pitch_tokenizer.py -v` 全件PASS |
| M1-7 | `compute_scl_loss()` 実装 | RQ-04 | NT-Xent対称版。`g_a, g_b: (K_s, D)`。L2正規化 → コサイン類似度 → `F.cross_entropy` 双方向平均。`tau=0.07` | `losses.py` (SCL) | `g_a==g_b` で loss≈0、出力スカラー |
| M1-8 | `compute_fcl_loss()` 実装 | RQ-04 | ソフトラベル対照学習。`f_a, f_b: (B,L,D)`, `Y: (B,L,L)` where `Y∈{+1,0}`。L2正規化 → 類似度 → `-(valid*Y*S).sum()/n_valid` | `losses.py` (FCL) | `Y` 全ゼロで loss==0、NaN/Infなし |
| M1-9 | `compute_svt_loss()` 実装 | RQ-04 | 3成分: L_CE(パディング除外CE) + L_seg(境界遷移損失) + L_dur(ソフトデュレーション損失)。λ_seg=3.0, λ_dur=5.0, δ=0.5 | `losses.py` (SVT) | 完全正解で `l_ce≈0`、`lambda_seg=0,lambda_dur=0` で `total==l_ce` |
| M1-10 | 統合損失 `compute_total_loss()` 実装 | RQ-04 | `L_total = 0.5*L_CL + 0.5*L_SVT + 0.3*L_mask`。`L_CL = 1.0*L_SCL + 0.1*L_FCL`。各λを引数で受取可能 | `losses.py` (統合) | 全成分NaN/Infなし、λ=0で対応成分消滅 |
| M1-11 | losses.py 単体テスト | RQ-04 | `tests/test_losses.py` 作成。M1-7〜10の受入基準を網羅 | `tests/test_losses.py` | `uv run pytest tests/test_losses.py -v` 全件PASS |
| M1-12 | 音響トークン抽出 | RQ-05 | `preprocess.py` 新規作成。Amphion Codec で `wav_24k → acoustic_tokens (T_a, 12)`。`maskgct_utils.py` パターンに準拠 | `preprocess.py` (音響) | shape `(T_a, 12)`, dtype `torch.long`, 値 `[0, 1023]` |
| M1-13 | セマンティックトークン抽出 | RQ-05 | w2v-bert-2.0 第17層 → z-score正規化 → VQ。50Hz→75Hzアップサンプリング | `preprocess.py` (セマンティック) | shape `(T_a,)` で音響トークンと同長 |
| M1-14 | F0抽出とピッチトークン化 | RQ-05 | pyworld DIO+StoneMask (24kHz, 5ms) → 200Hz F0 → `quantize_f0(f0, T_a)` で75Hz基準 | `preprocess.py` (F0) | shape `(T_a,)`, 値 `[0, 128]`, 無声→0 |
| M1-15 | 音素ID列生成 | RQ-05 | `pypinyin` でテキスト→ピンイン→音素ID。MaskGCTの G2P 流用 | `preprocess.py` (G2P) | 未知文字でクラッシュしない |
| M1-16 | `.pt` ファイル保存関数 | RQ-05 | 全フィールドを辞書にまとめて `torch.save`。長さ整合チェック含む | `save_preprocessed()` | 保存→ロードで全テンソル一致、長さ不整合で `ValueError` |
| M1-17 | バッチ前処理スクリプト | RQ-05 | ディレクトリ走査 → 各音声に M1-12〜16 適用。失敗スキップ、`tqdm` 進捗、JSON メタデータ出力 | `run_preprocess()` | 5ファイルテストで全処理完走、破損ファイル混入でもスキップ可能 |
| M1-18 | preprocess.py 統合テスト | RQ-05 | 合成正弦波（440Hz, 2秒）で M1-12〜16 を通し検証。モデル依存テストは `@pytest.mark.slow` | `tests/test_preprocess.py` | `uv run pytest tests/test_preprocess.py -v -m "not slow"` 全件PASS |

### 依存関係（タスク間）

```
並列グループA（完全独立）: M1-1 → M1-2 → M1-3 → M1-4 → M1-5 → M1-6
並列グループB（完全独立）: M1-7, M1-8, M1-9 → M1-10 → M1-11
グループC（A + M0-1 完了後）: M1-12, M1-13, M1-14, M1-15 → M1-16 → M1-17 → M1-18
```

### 完了判定基準

- [ ] `uv run pytest tests/test_pitch_tokenizer.py -v` 全件PASS
- [ ] `uv run pytest tests/test_losses.py -v` 全件PASS
- [ ] `uv run pytest tests/test_preprocess.py -v -m "not slow"` 全件PASS
- [ ] `quantize_f0([440.0], 1)` → `tensor([69])` をインタラクティブに確認
- [ ] `compute_scl_loss(g, g)` ≈ 0 を確認
- [ ] 合成正弦波1ファイルに対して `.pt` ファイルがエラーなく出力される

### リスクと対策

| リスク | 影響度 | 対策 |
|---|---|---|
| `models/codec` が sparse-checkout 後に不在 | 高 | M0-1 を最初に実施。不在時は `hf_hub_download` で別途取得 |
| `pyworld` ビルドエラー | 中 | `librosa.yin` (crepe代替) に切り替え可能な設計にする |
| 50Hz→75Hz アップサンプリングで長さズレ | 中 | `repeat_interleave` + 端数トリミングで `T_a` と厳密一致 |
| `loss_t()` のインプレース変更が pitch_emb の勾配グラフを壊す | 中 | RQ-03 実装時に `cond.clone()` を挟む設計を引き継ぐ |

---

## M2: コアモジュール実装

**目標**: CoMelSingerの中核となる3モジュール（SVTModule、Dataset/DataLoader、CoMelSinger_S2A）を実装し、単体テストで正確な入出力形状と数値挙動を検証する。

**前提条件**: M1完了（losses.py, pitch_tokenizer.py, preprocess.py 実装済み、前処理済みデータが `tokens/*.pt` 形式で存在）

**推定工数**: 10-13日

### タスク一覧

| ID | タスク | 対応RQ | 詳細 | 成果物 | 受入基準 |
|---|---|---|---|---|---|
| M2-1 | SVTModule クラス骨格 | RQ-02 | `svt_module.py` に `SVTModule(nn.Module)` を定義。パラメータ: `num_codebooks=12, codebook_size=1024, codebook_embed_dim=64, hidden_size=512, num_layers=4, num_heads=8, pitch_vocab_size=129, dropout=0.1` | `svt_module.py` | `SVTModule()` がエラーなくインスタンス化。パラメータ数 ～2M |
| M2-2 | `codebook_embs` + `input_proj` 実装 | RQ-02 | `nn.ModuleList([nn.Embedding(1024, 64) * 12])` + `nn.Linear(768, 512)`。`(B,L,12)` → 各Embed → `(B,L,768)` → Linear `(B,L,512)` | `svt_module.py` | `model._embed(torch.zeros(2,50,12).long())` → shape `(2,50,512)` |
| M2-3 | `pos_enc` + `transformer` 実装 | RQ-02 | `SinusoidalPosEmb(512)` + `nn.TransformerEncoder(4層, norm_first=True)`。`src_key_padding_mask` でパディングマスク | `svt_module.py` | 入力 `(2,50,512)` → 出力 `(2,50,512)` |
| M2-4 | `pitch_head` + `forward` メソッド | RQ-02 | `nn.Linear(512, 129)`。`forward()` → `{"logits": (B,L,129), "probs": (B,L,129)}` | `svt_module.py` | `probs.sum(-1).allclose(torch.ones(2,50))` |
| M2-5 | `freeze` / `unfreeze` ユーティリティ | RQ-02 | `freeze()`: `requires_grad_(False)` + `eval()`。S2A学習時に完全停止できること | `svt_module.py` | `freeze()` 後に `any(p.requires_grad for p in model.parameters())` が `False` |
| M2-6 | SVTModule の損失統合テスト | RQ-02 | M1の `compute_svt_loss` と組み合わせた `loss.backward()` が完走すること | テストコード | dummy batch で loss が有限値、NaN/Inf なし |
| M2-7 | `CoMelSingerDataset` クラス定義 | RQ-06 | `dataset.py` に Dataset 定義。`tokens/{uid}.pt` 読込、`acoustic_tokens` を `(12,T)` に転置 | `dataset.py` | `len(ds) > 0`, `ds[0]["acoustic_tokens"].shape[0] == 12` |
| M2-8 | `BalancedSpeakerSampler` 実装 | RQ-06 | バッチ毎に先頭 `k_s=8` サンプルが同一話者ペアを含むよう保証するサンプラー | `dataset.py` | バッチ内 `speaker_ids[:k_s]` に同一話者が2件以上 |
| M2-9 | `comelsinger_collate_fn` 実装 | RQ-06 | 可変長パディング。`acoustic_tokens: (B,12,T_max)`, `semantic_tokens: (B,T_max)`, `pitch_tokens: (B,T_max)`, `attention_mask: (B,T_max)` | `dataset.py` | batch=4 で shape が仕様通り |
| M2-10 | DataLoader 統合動作確認 | RQ-06 | `DataLoader(ds, batch_sampler=..., collate_fn=..., num_workers=2)` で1周イテレーション | テストコード | 1バッチ取得が60秒以内、全テンソル形状が仕様通り |
| M2-11 | `CoMelSinger_S2A` クラス骨格 | RQ-03 | `MaskGCT_S2A` を継承。`pitch_vocab_size=129, temperature=0.07`。ハイパーパラメータ: λ_cl=0.5, λ_scl=1.0, λ_fcl=0.1, λ_svt=0.5, λ_mask=0.3 | `comelsinger_s2a.py` | `isinstance(model, MaskGCT_S2A)` が `True` |
| M2-12 | `pitch_emb` + `forward` オーバーライド | RQ-03 | `pitch_emb = nn.Embedding(129, 1024)`。`cond = cond_emb(cond_code) + pitch_emb(pitch_tokens)` → `compute_loss(x0, x_mask, cond)` | `comelsinger_s2a.py` | `pitch_tokens=None` でも動作（後方互換） |
| M2-13 | 推論用 `get_cond` ヘルパー | RQ-03 | `get_cond(cond_code, pitch_tokens)` → `(B,T,1024)` cond テンソル生成。推論パイプラインからの呼び出し用 | `comelsinger_s2a.py` | `model.get_cond(code(1,100), pitch(1,100)).shape == (1,100,1024)` |
| M2-14 | マスク損失計算の統合確認 | RQ-03 | 親クラスの `compute_loss → loss_t → forward_diffusion` が pitch_emb 追加後も正常動作 | テストコード | `L_mask.item()` が有限値、`pitch_emb.weight.grad is not None` after backward |
| M2-15 | LoRA 適用後の動作確認 | RQ-03 | `LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], modules_to_save=["pitch_emb"])` 適用後のパラメータ比率と forward 確認 | テストスクリプト | 学習可能パラメータ 4-6%（目標 ≈4.83%）、LoRA+pitch_emb にのみ勾配 |

### 依存関係（タスク間）

```
並列グループA: M2-1 → M2-2 → M2-3 → M2-4 → M2-5 → M2-6
並列グループB: M2-7 → M2-8 → M2-9 → M2-10
グループC（グループA完了後）: M2-11 → M2-12 → M2-13 → M2-14 → M2-15

グループA と グループB は完全並列
```

### 完了判定基準

- [ ] SVTModule `(B=2, L=50, 12)` 入力で `logits.shape == (2,50,129)` かつ `probs.sum(-1) ≈ 1.0`
- [ ] `freeze()` 後に全パラメータの `requires_grad == False`
- [ ] `CoMelSingerDataset[0]` の `acoustic_tokens.shape[0] == 12`
- [ ] `BalancedSpeakerSampler` バッチに同一話者ペアが存在
- [ ] `CoMelSinger_S2A(pitch_tokens=None)` が後方互換で動作
- [ ] LoRA 適用後パラメータ比率 4-6%
- [ ] 全タスクで `loss.backward()` 後に NaN/Inf なし

### リスクと対策

| リスク | 影響度 | 対策 |
|---|---|---|
| `SinusoidalPosEmb` が 2D フレームシーケンスに非対応 | 高 | `pos_emb(torch.arange(L))` で位置エンコーディングを別生成しブロードキャスト加算 |
| `loss_t()` のインプレース `cond +=` が勾配グラフを壊す | 高 | `forward` で `cond.clone()` してから `compute_loss` に渡す |
| `BalancedSpeakerSampler` で話者数 < k_s の場合にエラー | 低 | ガード: `assert len(speakers) >= k_s // 2`、不足時はランダムサンプリングにフォールバック |
| LoRA `modules_to_save=["pitch_emb"]` と PEFT 保存機構の競合 | 中 | `save_pretrained()` → `from_pretrained()` の往復テストで確認 |

---

## M3: 学習パイプライン構築

**目標**: SVT独立学習（train_svt.py）とS2Aファインチューニング（train_s2a.py）の2本の学習スクリプトを実装し、チェックポイントを生成する。

**前提条件**: M2完了（全モジュール実装済み・単体テスト通過）

**推定工数**: 7-10日

### タスク一覧

| ID | タスク | 対応RQ | 詳細 | 成果物 | 受入基準 |
|---|---|---|---|---|---|
| M3-1 | 設定ファイル作成 | RQ-07 | `configs/svt_train.yaml` (lr=1e-5, 50Kステップ, batch=32, fp32) と `configs/s2a_train.yaml` (lr=1e-5, 100エポック, batch=32, bf16, k_s=8) を作成 | 設定ファイル2本 | パースエラーなし、必須キー全存在 |
| M3-2 | DataLoader検証 | RQ-06 | DataLoader のバッチ key/shape/dtype を検証。BalancedSpeakerSampler が同一話者ペアを保証することを確認 | 動作ログ | バッチ形状アサーション全通過 |
| M3-3 | `train_svt.py` スケルトン | RQ-07a | argparse、設定読込、seed=42、AdamW+CosineAnnealingLR初期化、チェックポイントディレクトリ作成 | `train_svt.py` 骨格 | `uv run python train_svt.py --config configs/svt_train.yaml --dry-run` が0ステップ動作 |
| M3-4 | train_svt: 学習ループ | RQ-07a | 50Kステップループ。forward → compute_svt_loss → backward → clip_grad(1.0) → step | `train_svt.py` ループ完成 | GPU×1で1ステップ実行、loss非NaN/Inf |
| M3-5 | train_svt: ロギング・チェックポイント | RQ-07a | TensorBoard で各損失成分を記録。1000ステップ毎にval F1計算＋チェックポイント保存。ベストF1更新時に `svt_best.pt` | チェックポイント群、TBログ | 1000ステップ後に `svt_best.pt` が存在 |
| M3-6 | train_svt: validation loop | RQ-07a | フレームレベル F1 計算関数 `evaluate_svt(model, val_loader)` | 評価関数 | val F1 が 0-1 の正常値 |
| M3-7 | train_svt: 小規模動作確認 | RQ-07a | 1000ステップ縮小実験（batch=4, GPU×1） | 実験ログ | loss_total が初期値の80%以下に低下 |
| M3-8 | `train_s2a.py` スケルトン | RQ-07b | Accelerate(bf16, DDP find_unused_params=True)、CoMelSinger_S2A構築、事前学習重み読込(strict=False)、LoRA適用、frozen SVT読込、AdamW+逆平方根スケジューラ | `train_s2a.py` 骨格 | `uv run python train_s2a.py --dry-run` が0エポック動作、trainable ≈4.8% |
| M3-9 | Alg.1 Step 1-2: バッチ分割 | RQ-07b | バッチB(K=32) → `S_A_s[:8]` (SCL用) + `S_A_f[8:]` (FCL+mask用) | バッチ分割コード | K_s=8, K-K_s=24 のshapeが正しい |
| M3-10 | Alg.1 Step 3: ピッチ摂動 | RQ-07b | `pitch_perturbation()`: zero_prob=0.5でトークン値をランダムシフト。摂動バッチB'構築 | ピッチ摂動コード | 摂動後のshape不変、値が一部異なる |
| M3-11 | Alg.1 Step 4-6: PromptGen | RQ-07b | 同一話者サンプルからランダムプロンプト切出。B用/B'用に異なるプロンプト生成 | `prompt_gen()` 関数 | prompts_a ≠ prompts_b、同一話者から生成 |
| M3-12 | Alg.1 Step 7-8: S2A forward | RQ-07b | `cond_B = cond_emb + pitch_emb(original)`、`cond_Bp = cond_emb + pitch_emb(perturbed)` | forward パスコード | cond_B/cond_Bp の shape が `(32,T,1024)` |
| M3-13 | Alg.1 Step 9-11: SCL損失 | RQ-07b | AvgPool → `g_a/g_b (K_s,1024)` → `compute_scl_loss(g_a, g_b, tau=0.07)` | SCL計算コード | L_SCL スカラー、NaN/Infなし |
| M3-14 | Alg.1 Step 12-13: FCL損失 | RQ-07b | `f_a/f_b = cond[K_s:]` → ソフトラベル行列Y → `compute_fcl_loss(f_a, f_b, Y)` | FCL計算コード | L_FCL スカラー、NaN/Infなし |
| M3-15 | Alg.1 Step 14: L_CL統合 | RQ-07b | `L_CL = 1.0 * L_SCL + 0.1 * L_FCL` | L_CL計算コード | 2成分が正しく合算 |
| M3-16 | Alg.1 Step 15-16: L_mask | RQ-07b | `s2a_model.compute_loss()` → マスク位置での `F.cross_entropy` | L_mask計算コード | L_mask スカラー、NaN/Infなし |
| M3-17 | Alg.1 Step 17-19: L_SVT (StopGrad) | RQ-07b | `with torch.no_grad(): svt_out = svt_model(detach)` → `compute_svt_loss(svt_out["probs"], ...)` | L_SVT計算コード | L_SVT スカラー、SVT重みのgradがNone |
| M3-18 | Alg.1 Step 20-21: 統合損失・更新 | RQ-07b | `L_total = 0.5*L_CL + 0.5*L_SVT + 0.3*L_mask` → backward → clip → step | 統合損失・更新コード | 1イテレーション後にLoRAパラメータのgradが非None |
| M3-19 | train_s2a: ロギング | RQ-07b | TensorBoard で loss_total/mask/scl/fcl/svt/lr を記録。10エポック毎にval指標 | TBログ | 10エポック後に5種損失カーブ確認可能 |
| M3-20 | train_s2a: チェックポイント保存 | RQ-07b | 10エポック毎に `s2a_ckpt_epoch{N}.pt`（LoRA差分）。val F0-RMSE最小更新時に `s2a_best.pt` | チェックポイント群 | `load_state_dict(strict=False)` でロード成功 |
| M3-21 | train_s2a: 小規模動作確認 | RQ-07b | 5エポック・batch=4・GPU×1 の縮小実験 | 実験ログ | 全5損失成分がNaN/Infなしで推移 |
| M3-22 | 設定値クロスチェック | RQ-07 | 要件定義書12の修正値（λ_SCL=1.0, λ_FCL=0.1, λ_SVT=0.5, K_s=8）が設定・コードに正しく反映されているか検証 | チェックリスト | 要件定義書12と実装値が完全一致 |

### 依存関係（タスク間）

```
M3-1（設定YAML）→ M3-3（svt骨格）→ M3-4（ループ）→ M3-5/M3-6 並列 → M3-7

M3-7（svt完成）→ M3-8（s2a骨格）→ M3-9 → M3-10 → M3-11 → M3-12
                  → M3-13/M3-14/M3-16/M3-17 並列 → M3-15 → M3-18
                  → M3-19 → M3-20 → M3-21 → M3-22
```

### 完了判定基準

- [ ] `uv run python train_svt.py` が1000ステップ走り loss が初期値の80%以下
- [ ] `svt_best.pt` が生成され `SVTModule` にロード可能
- [ ] `uv run python train_s2a.py` が5エポック走り全5損失成分が正常値
- [ ] SVT重みが学習後も変化していない（frozen確認）
- [ ] `s2a_best.pt` が生成され `CoMelSinger_S2A` にロード可能
- [ ] 要件定義書12のハイパーパラメータが設定・コードで一致
- [ ] TensorBoard で両学習の損失カーブが可視化可能
- [ ] `print_trainable_parameters()` で ≈4.8%

### リスクと対策

| リスク | 影響度 | 対策 |
|---|---|---|
| `find_unused_parameters=True` 指定漏れで DDP エラー | 高 | M3-8 で最初から DDPKwargs に含める |
| acoustic_tokens の次元順不一致 (B,12,T) vs (B,T,12) | 高 | M3-2 でバッチ shape を assert 確認、permute の要否を確定 |
| `modules_to_save=["pitch_emb"]` 未指定で pitch_emb がフリーズ | 高 | M3-8 で LoraConfig に含める、M3-22 で requires_grad=True を確認 |
| BF16 での SCL 数値不安定 | 中 | L_SCL 計算前に `g_a.float()` で FP32 キャスト |
| 逆平方根スケジューラが PyTorch 標準にない | 低 | `LambdaLR(optimizer, lr_lambda=lambda step: 1/math.sqrt(max(step, 1)))` で自前実装 |

---

## M4: 推論・評価システム

**目標**: 学習済みモデルを用いた推論パイプラインを実装し、論文 Table II/III/V/VI の再現に必要な全評価指標を測定できる評価基盤を構築する。

**前提条件**: M3完了（SVT学習済み + S2A LoRA fine-tuning 済みチェックポイント存在）

**推定工数**: 8-11日

### タスク一覧

| ID | タスク | 対応RQ | 詳細 | 成果物 | 受入基準 |
|---|---|---|---|---|---|
| M4-1 | 推論パイプライン基盤 | RQ-08 | `MaskGCT_Inference_Pipeline` を継承した `CoMelSinger_Inference_Pipeline` クラス。`pitch_tokenizer` をコンストラクタに追加 | `comelsinger_inference.py` | import・インスタンス化がエラーなし |
| M4-2 | `comelsinger_inference()` メソッド | RQ-08 | 楽譜入力 → tokenize_score → T2S → S2A第1層(`[25]`) → S2A全層(`[25,10,1,1,1,1,1,1,1,1,1,1]`) → vq2emb → codec_decoder の5ステップ | `comelsinger_inference()` | 任意の参照音声・楽譜入力で24kHz WAV出力 |
| M4-3 | LoRAマージ対応ロード | RQ-08 | LoRA チェックポイントロード + pitch_emb ロード。1layer/full 両モデルへ適用 | ロードユーティリティ | 学習済みチェックポイントがロード成功 |
| M4-4 | バッチ推論CLIスクリプト | RQ-08 | テストセット（Seen/Unseen 各50発話）一括推論。`--config`, `--testset`, `--output_dir` オプション | `run_inference.py` | 100発話の推論完走、WAV保存 |
| M4-5 | MCD 実装 | RQ-09 | メルケプストラム歪み。DTWアライメント。論文ターゲット: 4.17 dB | `evaluate.py` `compute_mcd()` | Seen 50発話のMCD平均値が出力される |
| M4-6 | F0-RMSE 実装 | RQ-09 | pyworld F0抽出 + 無声除外RMSE。論文ターゲット: 0.042 | `evaluate.py` `compute_f0_rmse()` | Seen 50発話のF0-RMSE平均値が出力される |
| M4-7 | SingMOS 実装 | RQ-09 | South-Twilight/SingMOS 導入。参照不要歌唱品質スコア。Seen目標: 4.32 | `evaluate.py` `compute_singmos()` | Seen/Unseen両セットでスコア出力 |
| M4-8 | SECS 実装 | RQ-09 | wavlm-base-plus でスピーカー埋め込みコサイン類似度。論文ターゲット: Seen 0.912 | `evaluate.py` `compute_secs()` | Seen/Unseen両セットでSECS出力 |
| M4-9 | SVT F1 評価 | RQ-09 | 合成音声→音響トークン再抽出→frozen SVT→ピッチ予測→GTとのフレームレベルF1。目標: 0.711 | `evaluate.py` `compute_svt_f1()` | SVT F1スコア出力 |
| M4-10 | Ablation設定ファイル整備 | RQ-09 | Table V再現用6条件: full/wo_cl/wo_scl/wo_fcl/wo_svt/wo_cl_svt の各YAML | `configs/ablation/*.yaml` (6本) | 各ファイルがロード可能、λ値が下表と一致 |
| M4-11 | Ablation結果集計スクリプト | RQ-09 | 6条件 × 自動評価指標のクロス集計表生成 | `ablation_summary.py`、CSV出力 | Table V形式のMarkdown表が自動生成 |
| M4-12 | 主観評価用サンプル生成ガイド | RQ-09 | MOS-Q/MOS-N/SMOS（20名・5段階）の手順書・信頼区間計算スクリプト | `docs/subjective_eval_guide.md` | 評価実施手順と計算コードが揃う |
| M4-13 | 評価結果統合レポート | RQ-09 | 自動評価全指標（MCD/F0-RMSE/SingMOS/SECS/SVT-F1）のJSON+Markdown表。論文Table II/III形式 | `evaluate.py --report` | 1コマンドでSeen/Unseen全指標出力 |

### Ablation設定値（Table V再現用）

| 構成 | λ_CL | λ_SCL | λ_FCL | λ_SVT | 備考 |
|---|---|---|---|---|---|
| full | 0.5 | 1.0 | 0.1 | 0.5 | 提案手法フル構成 |
| wo_cl | 0 | - | - | 0.5 | CL無効 |
| wo_scl | 0.5 | 0 | 0.1 | 0.5 | SCLのみ除去 |
| wo_fcl | 0.5 | 1.0 | 0 | 0.5 | FCLのみ除去 |
| wo_svt | 0.5 | 1.0 | 0.1 | 0 | SVT監督なし |
| wo_cl_svt | 0 | - | - | 0 | ベースライン相当 |

### 依存関係（タスク間）

```
M4-1 → M4-2 → M4-3 → M4-4
                        ├── M4-5, M4-6, M4-7, M4-8, M4-9 （全て並列実行可能）
                        │     └── M4-11（Ablation集計）→ M4-13（統合レポート）
                        ├── M4-10（Ablation設定ファイル）← M4-1と並列実施可能
                        └── M4-12（主観評価ガイド）← M4-4完了後すぐ開始可能
```

### 完了判定基準

- [ ] Seen/Unseen各50発話のWAVファイルが全数生成される
- [ ] MCD ≤ 4.67 dB（論文4.17 ± 0.5）
- [ ] F0-RMSE < 0.084（論文0.042の2倍以内）
- [ ] SingMOS ≥ 4.0（論文 4.32）
- [ ] SECS ≥ 0.85（論文 0.912）
- [ ] SVT F1 ≥ 0.6（論文 0.711）
- [ ] `wo_cl_svt`（ベースライン）より `full` の全指標が優位
- [ ] Ablation 6条件の集計表が自動生成される
- [ ] 主観評価実施手順書が整備されている

### リスクと対策

| リスク | 影響度 | 対策 |
|---|---|---|
| SingMOS 外部リポジトリが非公開/破損 | 高 | PyPI版確認。不可時は UTMOS 等の代替MOS推定器を使用 |
| F0-RMSE の単位不一致（Hz vs semitones） | 中 | 両単位で計算し論文値に近い方を採用 |
| Ablation 6条件の学習チェックポイントが揃わない | 中 | `full` 1条件で先行検証、Ablation学習完了後に6条件比較 |
| 主観評価（20名確保）はM4スコープ外 | 高 | 自動評価で再現性を先行検証。主観評価はオプション扱い |
| テストセット未整備（Seen/Unseen 50発話） | 中 | 論文に従い話者バランスを考慮したシード固定サンプリングを実装 |

---

## 付録: タスク総数サマリー

| マイルストーン | タスク数 | 推定工数 | 対応RQ |
|---|---|---|---|
| M0: 環境構築 | 13 | 1-2日 | - |
| M1: 基盤モジュール | 18 | 5-7日 | RQ-01, RQ-04, RQ-05 |
| M2: コアモジュール | 15 | 10-13日 | RQ-02, RQ-06, RQ-03 |
| M3: 学習パイプライン | 22 | 7-10日 | RQ-07a, RQ-07b |
| M4: 推論・評価 | 13 | 8-11日 | RQ-08, RQ-09 |
| **合計** | **81** | **31-43日** | |

---

## 付録: 実装注意事項

### Python実行環境
- **全てのPythonコマンドは `uv run python` 経由で実行する**
- 依存追加は `uv add <package>` のみ
- `pip install`, `uv pip`, 素の `python` コマンドは使用禁止

### loss_t() のインプレース変更への対応
`MaskGCT_S2A.loss_t()` は `cond += self.layer_emb(mask_layer).unsqueeze(1)` でインプレース加算を行う。`CoMelSinger_S2A.forward()` で pitch_emb を加算した cond を渡す際、この変更が FCL 計算用テンソルに影響しないよう `cond.clone()` を挟む設計にすること。

### BF16精度での数値安定性
SCL 損失計算時のコサイン類似度は BF16 でアンダーフローが起きやすい。`g_a.float()` で FP32 にキャストしてから計算し、結果を BF16 に戻す。

### acoustic_tokens の次元順
- DataLoader 出力: `(B, 12, T_max)`
- SVTModule 入力: `(B, T, 12)` → `permute(0, 2, 1)` が必要
- MaskGCT_S2A 入力: `(B, T, 12)`
- 学習スクリプトで統一的に変換すること
