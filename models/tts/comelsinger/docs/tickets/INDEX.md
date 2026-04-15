# CoMelSinger チケット管理インデックス

> **マイルストーン計画書**: [13_milestones.md](../13_milestones.md)
> **要件定義書**: [12_requirements_specification.md](../12_requirements_specification.md)

---

## 進捗サマリー

| Phase | マイルストーン | チケット数 | 完了 | 進捗 |
|---|---|---|---|---|
| M0 | 環境構築 | 13 | 0 | 0% |
| M1 | 基盤モジュール | 18 | 0 | 0% |
| M2 | コアモジュール | 15 | 0 | 0% |
| M3 | 学習パイプライン | 22 | 0 | 0% |
| M4 | 推論・評価 | 13 | 0 | 0% |
| **合計** | | **81** | **0** | **0%** |

---

## M0: 環境構築 (1-2日)

| チケット | タスク | 状態 | 依存 |
|---|---|---|---|
| [M0-01](M0/M0-01_sparse_checkout_codec.md) | sparse-checkout に models/codec 追加 | TODO | - |
| [M0-02](M0/M0-02_uv_init.md) | uv init でプロジェクト初期化 | TODO | - |
| [M0-03](M0/M0-03_pytorch_install.md) | PyTorch 系ライブラリインストール | TODO | M0-02 |
| [M0-04](M0/M0-04_ml_libs_install.md) | ML 系ライブラリインストール | TODO | M0-03 |
| [M0-05](M0/M0-05_audio_libs_install.md) | 音声処理系ライブラリインストール | TODO | M0-03 |
| [M0-06](M0/M0-06_amphion_deps_install.md) | Amphion 固有依存インストール | TODO | M0-03 |
| [M0-07](M0/M0-07_download_maskgct_t2s.md) | 事前学習済みモデル: MaskGCT-T2S | TODO | - |
| [M0-08](M0/M0-08_download_maskgct_s2a.md) | 事前学習済みモデル: MaskGCT-S2A | TODO | - |
| [M0-09](M0/M0-09_download_w2v_bert.md) | 事前学習済みモデル: w2v-bert-2.0 | TODO | - |
| [M0-10](M0/M0-10_download_amphion_codec.md) | 事前学習済みモデル: Amphion Codec | TODO | M0-01 |
| [M0-11](M0/M0-11_verify_w2v_stats.md) | wav2vec2bert_stats.pt 配置確認 | TODO | - |
| [M0-12](M0/M0-12_smoke_test_maskgct.md) | MaskGCT 既存コード動作確認 | TODO | M0-01〜M0-06 |
| [M0-13](M0/M0-13_verify_comelsinger_pkg.md) | comelsinger パッケージ初期化確認 | TODO | M0-12 |

## M1: 基盤モジュール (5-7日)

| チケット | タスク | 状態 | 依存 |
|---|---|---|---|
| [M1-01](M1/M1-01_pitch_tokenizer_skeleton.md) | PitchTokenizer クラス骨格 | TODO | M0 |
| [M1-02](M1/M1-02_quantize_f0.md) | quantize_f0() 実装 | TODO | M1-01 |
| [M1-03](M1/M1-03_tokenize_score.md) | tokenize_score() 実装 | TODO | M1-02 |
| [M1-04](M1/M1-04_soft_label_matrix.md) | compute_soft_label_matrix() 実装 | TODO | M1-03 |
| [M1-05](M1/M1-05_decode_token_to_freq.md) | decode_token_to_freq() 実装 | TODO | M1-01 |
| [M1-06](M1/M1-06_test_pitch_tokenizer.md) | PitchTokenizer 単体テスト | TODO | M1-02〜M1-05 |
| [M1-07](M1/M1-07_scl_loss.md) | compute_scl_loss() 実装 | TODO | M0 |
| [M1-08](M1/M1-08_fcl_loss.md) | compute_fcl_loss() 実装 | TODO | M0 |
| [M1-09](M1/M1-09_svt_loss.md) | compute_svt_loss() 実装 | TODO | M0 |
| [M1-10](M1/M1-10_total_loss.md) | compute_total_loss() 実装 | TODO | M1-07〜M1-09 |
| [M1-11](M1/M1-11_test_losses.md) | losses.py 単体テスト | TODO | M1-07〜M1-10 |
| [M1-12](M1/M1-12_extract_acoustic_tokens.md) | 音響トークン抽出 | TODO | M0-01 |
| [M1-13](M1/M1-13_extract_semantic_tokens.md) | セマンティックトークン抽出 | TODO | M0-09 |
| [M1-14](M1/M1-14_extract_pitch_tokens.md) | F0 抽出とピッチトークン化 | TODO | M1-02 |
| [M1-15](M1/M1-15_extract_phone_ids.md) | 音素 ID 列生成 | TODO | M0 |
| [M1-16](M1/M1-16_save_preprocessed.md) | .pt ファイル保存関数 | TODO | M1-12〜M1-15 |
| [M1-17](M1/M1-17_batch_preprocess.md) | バッチ前処理スクリプト | TODO | M1-16 |
| [M1-18](M1/M1-18_test_preprocess.md) | preprocess.py 統合テスト | TODO | M1-17 |

## M2: コアモジュール (10-13日)

| チケット | タスク | 状態 | 依存 |
|---|---|---|---|
| [M2-01](M2/M2-01_svt_skeleton.md) | SVTModule クラス骨格 | TODO | M1 |
| [M2-02](M2/M2-02_svt_codebook_embed.md) | codebook_embs + input_proj 実装 | TODO | M2-01 |
| [M2-03](M2/M2-03_svt_transformer.md) | pos_enc + transformer 実装 | TODO | M2-02 |
| [M2-04](M2/M2-04_svt_forward.md) | pitch_head + forward メソッド | TODO | M2-03 |
| [M2-05](M2/M2-05_svt_freeze.md) | freeze/unfreeze ユーティリティ | TODO | M2-04 |
| [M2-06](M2/M2-06_svt_loss_integration.md) | SVTModule 損失統合テスト | TODO | M2-05 |
| [M2-07](M2/M2-07_dataset_class.md) | CoMelSingerDataset クラス定義 | TODO | M1 |
| [M2-08](M2/M2-08_balanced_sampler.md) | BalancedSpeakerSampler 実装 | TODO | M2-07 |
| [M2-09](M2/M2-09_collate_fn.md) | comelsinger_collate_fn 実装 | TODO | M2-08 |
| [M2-10](M2/M2-10_dataloader_integration.md) | DataLoader 統合動作確認 | TODO | M2-09 |
| [M2-11](M2/M2-11_s2a_skeleton.md) | CoMelSinger_S2A クラス骨格 | TODO | M1 |
| [M2-12](M2/M2-12_s2a_pitch_emb.md) | pitch_emb + forward オーバーライド | TODO | M2-11 |
| [M2-13](M2/M2-13_s2a_get_cond.md) | 推論用 get_cond ヘルパー | TODO | M2-12 |
| [M2-14](M2/M2-14_s2a_mask_loss.md) | マスク損失計算統合確認 | TODO | M2-13 |
| [M2-15](M2/M2-15_s2a_lora.md) | LoRA 適用後動作確認 | TODO | M2-14 |

## M3: 学習パイプライン (7-10日)

| チケット | タスク | 状態 | 依存 |
|---|---|---|---|
| [M3-01](M3/M3-01_config_files.md) | 設定ファイル作成 | TODO | M2 |
| [M3-02](M3/M3-02_dataloader_verify.md) | DataLoader 検証 | TODO | M2-10 |
| [M3-03](M3/M3-03_train_svt_skeleton.md) | train_svt.py スケルトン | TODO | M3-01 |
| [M3-04](M3/M3-04_train_svt_loop.md) | train_svt 学習ループ | TODO | M3-03 |
| [M3-05](M3/M3-05_train_svt_logging.md) | train_svt ロギング・チェックポイント | TODO | M3-04 |
| [M3-06](M3/M3-06_train_svt_validation.md) | train_svt validation loop | TODO | M3-04 |
| [M3-07](M3/M3-07_train_svt_smoke.md) | train_svt 小規模動作確認 | TODO | M3-05, M3-06 |
| [M3-08](M3/M3-08_train_s2a_skeleton.md) | train_s2a.py スケルトン | TODO | M3-07 |
| [M3-09](M3/M3-09_alg1_batch_split.md) | Alg.1 バッチ分割 | TODO | M3-08 |
| [M3-10](M3/M3-10_alg1_pitch_perturb.md) | Alg.1 ピッチ摂動 | TODO | M3-09 |
| [M3-11](M3/M3-11_alg1_prompt_gen.md) | Alg.1 PromptGen | TODO | M3-10 |
| [M3-12](M3/M3-12_alg1_s2a_forward.md) | Alg.1 S2A forward | TODO | M3-11 |
| [M3-13](M3/M3-13_alg1_scl.md) | Alg.1 SCL 損失 | TODO | M3-12 |
| [M3-14](M3/M3-14_alg1_fcl.md) | Alg.1 FCL 損失 | TODO | M3-12 |
| [M3-15](M3/M3-15_alg1_l_cl.md) | Alg.1 L_CL 統合 | TODO | M3-13, M3-14 |
| [M3-16](M3/M3-16_alg1_l_mask.md) | Alg.1 L_mask | TODO | M3-12 |
| [M3-17](M3/M3-17_alg1_l_svt.md) | Alg.1 L_SVT (StopGrad) | TODO | M3-12 |
| [M3-18](M3/M3-18_alg1_total_loss.md) | Alg.1 統合損失・更新 | TODO | M3-15〜M3-17 |
| [M3-19](M3/M3-19_train_s2a_logging.md) | train_s2a ロギング | TODO | M3-18 |
| [M3-20](M3/M3-20_train_s2a_checkpoint.md) | train_s2a チェックポイント | TODO | M3-19 |
| [M3-21](M3/M3-21_train_s2a_smoke.md) | train_s2a 小規模動作確認 | TODO | M3-20 |
| [M3-22](M3/M3-22_config_crosscheck.md) | 設定値クロスチェック | TODO | M3-21 |

## M4: 推論・評価 (8-11日)

| チケット | タスク | 状態 | 依存 |
|---|---|---|---|
| [M4-01](M4/M4-01_inference_pipeline.md) | 推論パイプライン基盤 | TODO | M3 |
| [M4-02](M4/M4-02_inference_method.md) | comelsinger_inference() メソッド | TODO | M4-01 |
| [M4-03](M4/M4-03_lora_load.md) | LoRA マージ対応ロード | TODO | M4-02 |
| [M4-04](M4/M4-04_batch_inference.md) | バッチ推論 CLI スクリプト | TODO | M4-03 |
| [M4-05](M4/M4-05_eval_mcd.md) | MCD 実装 | TODO | M4-04 |
| [M4-06](M4/M4-06_eval_f0_rmse.md) | F0-RMSE 実装 | TODO | M4-04 |
| [M4-07](M4/M4-07_eval_singmos.md) | SingMOS 実装 | TODO | M4-04 |
| [M4-08](M4/M4-08_eval_secs.md) | SECS 実装 | TODO | M4-04 |
| [M4-09](M4/M4-09_eval_svt_f1.md) | SVT F1 評価 | TODO | M4-04 |
| [M4-10](M4/M4-10_ablation_configs.md) | Ablation 設定ファイル整備 | TODO | M4-01 |
| [M4-11](M4/M4-11_ablation_summary.md) | Ablation 結果集計スクリプト | TODO | M4-05〜M4-09 |
| [M4-12](M4/M4-12_subjective_eval_guide.md) | 主観評価用サンプル生成ガイド | TODO | M4-04 |
| [M4-13](M4/M4-13_eval_report.md) | 評価結果統合レポート | TODO | M4-05〜M4-09 |
