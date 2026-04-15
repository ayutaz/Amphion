# M1-13: セマンティックトークン抽出 (extract_semantic_tokens)

> **マイルストーン**: [M1: 基盤モジュール](../../13_milestones.md#m1-基盤モジュール実装)
> **対応RQ**: RQ-05
> **依存チケット**: M0-09, M0-11
> **ブロックするチケット**: M1-16, M1-18
> **状態**: TODO

---

## 1. 目的とゴール

Wav2Vec2Bert の第17層隠れ状態を z-score 正規化し、RepCodec で VQ することでセマンティックトークン列を抽出する。50Hz 出力を 75Hz に nearest-neighbor アップサンプリングする。

**ゴール**: `semantic_tokens (T_a,)` long型、75Hz 基準で音響トークンと同長。

## 2. 実装する内容の詳細

```python
def extract_semantic_tokens(
    speech: torch.Tensor,           # (T,), 24kHz → 内部で16kHzリサンプル
    w2v_bert_model,                 # Wav2Vec2BertModel
    repcodec_model,                 # RepCodec VQ
    mean: torch.Tensor, var: torch.Tensor,  # z-score 統計
    target_len: int,                # T_a (75Hz 基準)
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    with torch.no_grad():
        outputs = w2v_bert_model(speech_16k.unsqueeze(0), output_hidden_states=True)
        hidden = outputs.hidden_states[17].squeeze(0)  # (T_50, D)
        hidden = (hidden - mean) / (var.sqrt() + 1e-8)
        # 50Hz → 75Hz nearest-neighbor
        hidden = hidden.unsqueeze(0).permute(0, 2, 1)  # (1, D, T_50)
        hidden = torch.nn.functional.interpolate(hidden, size=target_len, mode="nearest")
        hidden = hidden.permute(0, 2, 1).squeeze(0)     # (T_a, D)
        semantic_tokens = repcodec_model.encode(hidden)  # (T_a,)
    return semantic_tokens
```

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当 |
|---|---|---|
| 実装 | 1 | 関数実装・リサンプリングロジック |
| レビュー | 1 | RepCodec API・hidden layer インデックス確認 |

## 4. 提供範囲とテスト項目

### 4.2 ユニットテスト

- 50Hz→75Hz アップサンプリングの比率確認（`interpolate` 単体テスト）
- 出力 shape `(T_a,)`、dtype `torch.long`

### 4.3 E2Eテスト

- M1-18 統合テストで M1-12 と同一音声から得た `T_a` を `target_len` として渡し長さ整合を確認

## 5. 懸念事項とレビュー項目

### 5.1 懸念事項

- 第17層インデックスが 0-indexed vs 1-indexed で論文と不一致の可能性
- RepCodec の `encode` メソッド API が `maskgct_utils.py` と異なる可能性
- 24kHz→16kHz リサンプリングが必要（w2v-bert-2.0 は 16kHz 入力）

### 5.2 レビュー項目

- [ ] `output_hidden_states=True` が指定されているか
- [ ] `mean`/`var` の shape が hidden_dim と一致しているか

## 6. フェーズ振り返り: 一から作り直すとしたら

- **フレームレート統一**: 全モジュールを 75Hz に統一し、アップサンプリング変換を排除
- **特徴抽出器抽象化**: Whisper/Wav2Vec2Bert を `SemanticExtractor` プロトコルで統一
- **統計値自動計算**: 前処理パイプラインの一部として mean/var を自動計算・保存

## 7. 後続タスクへの連絡事項

- **M1-14**: 同じ `target_len` を渡してピッチトークンの長さを統一
- **M1-16**: `semantic_tokens` を `(T_a,)` shape のまま保存
