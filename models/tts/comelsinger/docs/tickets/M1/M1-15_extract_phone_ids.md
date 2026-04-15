# M1-15: 音素ID列生成 (extract_phone_ids)

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-05
> **依存チケット**: M0-05
> **ブロックするチケット**: M1-16, M1-18
> **状態**: TODO

---

## 1. 目的とゴール

pypinyin で中国語テキストをピンインに変換し、MaskGCT の G2P モジュールで音素 ID 列に変換する。

**ゴール**: 入力テキスト (str) → `phone_ids (T_ph,)` long型。未知文字は `<unk>` にフォールバック。

## 2. 実装する内容の詳細

```python
from pypinyin import lazy_pinyin, Style

def extract_phone_ids(
    text: str,
    phone2id: dict,         # {"a": 0, "b": 1, ..., "<unk>": N}
    language: str = "zh",
) -> torch.Tensor:
    pinyins = lazy_pinyin(text, style=Style.TONE3)
    # MaskGCT G2P で音素列に変換
    phones = g2p(pinyins, language=language)
    ids = [phone2id.get(p, phone2id.get("<unk>", 0)) for p in phones]
    return torch.tensor(ids, dtype=torch.long)
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当 |
|---|---|---|
| 実装 | 1 | 関数実装・G2P 流用 |
| レビュー | 1 | G2P 出力確認・ボキャブラリカバレッジ |

## 4. 提供範囲とテスト項目

### 4.2 ユニットテスト

- "你好" → dtype=long, dim=1, numel > 0
- 未知文字で `<unk>` ID にフォールバック（クラッシュしない）

### 4.3 E2Eテスト

- M1-18 で "你好世界" に対して phone_ids が正常生成されること

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- MaskGCT の `g2p_generation.py` が期待する入力形式を要確認
- pypinyin の多音字精度。歌唱テキストでは精度不足の場合カスタム辞書で補正
- `<unk>` トークンが多発する場合はボキャブラリカバレッジ不足

### 5.2 レビュー項目

- [ ] `phone2id` に `<unk>` キーが存在することを確認
- [ ] 空文字列入力でクラッシュしないか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **多言語対応**: `G2PBackend` 抽象クラスで中国語/日本語/英語を統一インターフェース
- **音節 ID 分離**: 音素 ID とは別に音節 ID を保持し、音符アライメントを明示管理
- **ボキャブラリ管理**: `tokenizers` ライブラリ形式で特殊トークンを統一

## 7. 後続タスクへの連絡事項

- **M1-16**: `phone_ids` を `(T_ph,)` shape で保存。`T_ph` は `T_a` と異なる長さ
- **T2S**: 音素 ID 列は T2S モデルの入力。MaskGCT_T2S の入力フォーマットとの整合を確認
