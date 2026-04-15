# M0-05: 音声処理系ライブラリインストール

> **マイルストーン**: [M0: 環境構築](../../13_milestones.md#m0-環境構築)
> **対応RQ**: -（環境構築）
> **依存チケット**: M0-03
> **ブロックするチケット**: M1-14, M1-15
> **状態**: TODO

---

## 1. 目的とゴール

CoMelSinger のデータ前処理パイプラインは以下の音声処理ライブラリを使用する:
- `pyworld`: F0 抽出 → ピッチトークン化
- `librosa`: 音声読み込み・特徴量抽出
- `pypinyin`: 中国語 G2P
- `mir_eval`: SVT F1 評価

## 2. 実装する内容の詳細

```bash
uv add "pyworld>=0.3.4" "librosa>=0.10.1" "pypinyin>=0.50.0" "mir_eval>=0.7"
```

### pyworld 動作確認

```python
import numpy as np
import pyworld as pw
sr = 24000
audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float64)
f0, sp, ap = pw.wav2world(audio, sr)
print('F0 mean (voiced):', f0[f0 > 0].mean())  # 440Hz 付近
```

### pypinyin 動作確認

```python
from pypinyin import pinyin, Style
result = pinyin("你好世界", style=Style.TONE3)
print('Pinyin:', result)  # [['ni3'], ['hao3'], ['shi4'], ['jie4']]
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当内容 |
|---|---|---|
| 実行エージェント | 1 | uv add 実行・動作確認 |

## 4. 提供範囲とテスト項目

### 4.1 提供範囲（スコープ）

**含むもの**: 4ライブラリのインストール・基礎動作確認

**含まないもの**: `crepe`（オプション）、データセット固有の前処理スクリプト（→ M1）

### 4.2 ユニットテスト

```bash
uv run python -c "
import numpy as np
import pyworld as pw
sr = 24000
audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float64)
f0, sp, ap = pw.wav2world(audio, sr)
assert f0.shape[0] > 0
from pypinyin import pinyin, Style
result = pinyin('你好', style=Style.TONE3)
assert len(result) == 2
print('PASS')
"
```

### 4.3 E2Eテスト

```bash
uv run python -c "
import numpy as np, soundfile as sf, librosa, tempfile, os
sr = 24000
audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr)).astype(np.float32)
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    sf.write(f.name, audio, sr)
    y, sr_loaded = librosa.load(f.name, sr=sr)
    os.unlink(f.name)
assert abs(len(y) - sr) < 100
print('PASS')
"
```

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- **pyworld のビルド失敗**: macOS で C 拡張がビルドできない場合がある。`uv add pyworld --no-binary pyworld` でソースビルドを試す。
- **pypinyin の多音字精度**: デフォルトでは文脈無視変換。M1-15 でカスタム辞書を検討。
- **pyworld は float64 を要求**: `float32` を渡すと無音扱いされる。

### 5.2 レビュー項目

- [ ] 4ライブラリ全てインポート成功
- [ ] `pyworld.wav2world` でダミー音声の F0 抽出が動作
- [ ] `pypinyin.pinyin` で中国語テキスト変換が動作

## 6. フェーズ振り返り: 一から作り直すとしたら

**F0 抽出器の選択**: 論文は pyworld を使用。一から設計するなら `PitchExtractor` 抽象クラスで pyworld/crepe/pyin を切り替え可能にする。

**中国語 G2P**: `pypinyin` をベースにカスタム辞書で補正する設計がバランスが良い。

## 7. 後続タスクへの連絡事項

- **M1-14**: `pyworld.wav2world` は `float64` の numpy 配列を要求。
- **M1-15**: `pypinyin.pinyin(text, style=Style.TONE3)` を使用。多音字精度は M1-15 で対処。
- pyworld のビルド状況を記録し CI/CD 環境で同様の問題が起きないよう `uv.lock` にコミットすること。
