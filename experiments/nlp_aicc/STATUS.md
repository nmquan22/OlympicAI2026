# Tiến độ thực nghiệm NLP

**54 thí nghiệm T4 thành công đã lưu bằng chứng; mục tiêu tối thiểu 42 (7/bài).** Số thí nghiệm khác số Saved Version; lượt lỗi không được tính. Tối đa hai Saved Version đang chạy.

| Bài | Đã hoàn tất | Tốt nhất trong điều kiện đã chạy |
|---|---:|---|
| [Moment Match](moment-match/IDEAS.md) | 12/7 | R07: public 0.6619, private 0.6162 Accuracy |
| [Lost Interpreter](lost-interpreter/IDEAS.md) | 10/7 | R08: public 14.1608, private 16.7920 MAE |
| [Word Lookups](word-lookups/IDEAS.md) | 9/7 | R08: public 0.9150, private 0.9171 Boundary F1 |
| [Inverse Translation](inverse-translation/IDEAS.md) | 9/7 | R09: public 100.0000, private 100.0000 Competition score |
| [Alchemy](alchemy/IDEAS.md) | 11/7 | R01: public 0.0285, private 0.0000 Accuracy |
| [Tally](tally/IDEAS.md) | 3/7 | R03: public 0.3530, private 0.3530 Entity-span micro F1 |

Accuracy/MAE trên validation, LM NLL proxy và official inverse score là các metric khác nhau. Không so điểm ngang giữa các bài; không gọi NLL là Boundary F1.

## Moment Match

| Run | Ý tưởng | Kết quả | Nhật ký |
|---|---|---|---|
| R01 | BPE + soft late interaction | 0.65439 accuracy; Kaggle 0.6121/0.5959 Accuracy (public/private) | [Đọc](moment-match/runs/R01/LEARNED.md) |
| R02 | Thêm Transformer ở mức token | 0.66383 accuracy; Kaggle 0.6406/0.6172 Accuracy (public/private) | [Đọc](moment-match/runs/R02/LEARNED.md) |
| R03 | Thêm positional Transformer ở mức sự kiện | 0.60434 accuracy; Kaggle 0.5898/0.5695 Accuracy (public/private) | [Đọc](moment-match/runs/R03/LEARNED.md) |
| R04 | IDF weighting cho alignment | 0.69122 accuracy; Kaggle 0.6436/0.6284 Accuracy (public/private) | [Đọc](moment-match/runs/R04/LEARNED.md) |
| R05 | Giảm nhiệt độ soft alignment | 0.66100 accuracy; Kaggle 0.6304/0.6060 Accuracy (public/private) | [Đọc](moment-match/runs/R05/LEARNED.md) |
| R06 | Chỉ giữ sự kiện trung tâm để đo vai trò ngữ cảnh | 0.49103 accuracy; Kaggle 0.4659/0.4690 Accuracy (public/private) | [Đọc](moment-match/runs/R06/LEARNED.md) |
| R07 | Kết hợp token attention + IDF, dựa trên R03/R04 | 0.66478 accuracy; Kaggle 0.6619/0.6162 Accuracy (public/private) | [Đọc](moment-match/runs/R07/LEARNED.md) |
| R08 | IDF + prior căn chỉnh đơn điệu khả vi | 0.68933 accuracy; Kaggle 0.6477/0.6253 Accuracy (public/private) | [Đọc](moment-match/runs/R08/LEARNED.md) |
| R09 | IDF + max alignment gần reference | 0.67705 accuracy; Kaggle 0.6467/0.6182 Accuracy (public/private) | [Đọc](moment-match/runs/R09/LEARNED.md) |
| R10 | BPE view + nguyên word-vector feature view được cấp | 0.64873 accuracy; Kaggle 0.6162/0.5969 Accuracy (public/private) | [Đọc](moment-match/runs/R10/LEARNED.md) |
| R11 | Event-level decomposable cross attention with compare and aggregate | 0.65722 accuracy; Kaggle 0.6304/0.6233 Accuracy (public/private) | [Đọc](moment-match/runs/R11/LEARNED.md) |
| R12 | Soft-DTW event alignment as a learned residual score | 0.69405 accuracy; Kaggle 0.6456/0.6253 Accuracy (public/private) | [Đọc](moment-match/runs/R12/LEARNED.md) |

## Lost Interpreter

| Run | Ý tưởng | Kết quả | Nhật ký |
|---|---|---|---|
| R01 | BPE Transformer + mean/max + MAE | 36.26026 MAE; Kaggle 36.9252/39.3060 MAE (public/private) | [Đọc](lost-interpreter/runs/R01/LEARNED.md) |
| R02 | Thêm separator lệnh và marker số | 37.94136 MAE; Kaggle 38.1944/40.5780 MAE (public/private) | [Đọc](lost-interpreter/runs/R02/LEARNED.md) |
| R03 | Nhánh thống kê số fit trên train | 37.66859 MAE; Kaggle 37.4260/40.1384 MAE (public/private) | [Đọc](lost-interpreter/runs/R03/LEARNED.md) |
| R04 | Learned query attention pooling | 37.77339 MAE; Kaggle 37.5484/40.3992 MAE (public/private) | [Đọc](lost-interpreter/runs/R04/LEARNED.md) |
| R05 | Auxiliary loss dự đoán số lệnh | 37.60918 MAE; Kaggle 37.4428/40.3316 MAE (public/private) | [Đọc](lost-interpreter/runs/R05/LEARNED.md) |
| R06 | Signed-log target để kiểm tra outlier | 38.02840 MAE; Kaggle 37.9180/40.7816 MAE (public/private) | [Đọc](lost-interpreter/runs/R06/LEARNED.md) |
| R07 | Kết hợp structure + numeric + query + auxiliary | 37.57629 MAE; Kaggle 37.0200/39.9584 MAE (public/private) | [Đọc](lost-interpreter/runs/R07/LEARNED.md) |
| R08 | BPE BiGRU + packed masking + SmoothL1 theo cùng holdout | 15.30885 MAE; Kaggle 14.1608/16.7920 MAE (public/private) | [Đọc](lost-interpreter/runs/R08/LEARNED.md) |
| R09 | Typed instruction BPE plus exact operand channels and recurrent execution | 15.81364 MAE; Kaggle 14.9744/17.8852 MAE (public/private) | [Đọc](lost-interpreter/runs/R09/LEARNED.md) |
| R10 | Typed instruction model with mixed easy/full curriculum | 16.31780 MAE; Kaggle 15.2480/18.4884 MAE (public/private) | [Đọc](lost-interpreter/runs/R10/LEARNED.md) |

## Word Lookups

| Run | Ý tưởng | Kết quả | Nhật ký |
|---|---|---|---|
| R01 | Transformer surprisal với offset-safe BPE | 6.10684 BPE_LM_NLL_proxy; Kaggle 0.7559/0.7560 Boundary F1 (public/private) | [Đọc](word-lookups/runs/R01/LEARNED.md) |
| R02 | Surprisal hai chiều | 5.80200 BPE_LM_NLL_proxy; Kaggle 0.8179/0.8214 Boundary F1 (public/private) | [Đọc](word-lookups/runs/R02/LEARNED.md) |
| R03 | Chuẩn hóa surprisal theo vị trí/độ dài | 6.10684 BPE_LM_NLL_proxy; Kaggle 0.7604/0.7594 Boundary F1 (public/private) | [Đọc](word-lookups/runs/R03/LEARNED.md) |
| R04 | Kết hợp cohesion n-gram học từ train | 6.10684 BPE_LM_NLL_proxy; Kaggle 0.8766/0.8775 Boundary F1 (public/private) | [Đọc](word-lookups/runs/R04/LEARNED.md) |
| R05 | DP với prior độ dài từ | 6.10684 BPE_LM_NLL_proxy; Kaggle 0.7578/0.7583 Boundary F1 (public/private) | [Đọc](word-lookups/runs/R05/LEARNED.md) |
| R06 | Punctuation-aware BMES constraints | 6.10684 BPE_LM_NLL_proxy; Kaggle 0.8196/0.8215 Boundary F1 (public/private) | [Đọc](word-lookups/runs/R06/LEARNED.md) |
| R07 | Kết hợp hai chiều + cohesion + DP | 5.80200 BPE_LM_NLL_proxy; Kaggle 0.9097/0.9113 Boundary F1 (public/private) | [Đọc](word-lookups/runs/R07/LEARNED.md) |
| R08 | Accessor-variety lexical spans supplement boundary surprise | 5.80200 BPE_LM_NLL_proxy; Kaggle 0.9150/0.9171 Boundary F1 (public/private) | [Đọc](word-lookups/runs/R08/LEARNED.md) |
| R09 | Accessor spans plus BPE-dropout regularized scratch LM | 5.78945 BPE_LM_NLL_proxy; Kaggle 0.9145/0.9168 Boundary F1 (public/private) | [Đọc](word-lookups/runs/R09/LEARNED.md) |

## Inverse Translation

| Run | Ý tưởng | Kết quả | Nhật ký |
|---|---|---|---|
| R01 | Probe streaming giữ top-32, margin search | 100.00000 official_score_0_100; Kaggle 100.0000/100.0000 Competition score (public/private) | [Đọc](inverse-translation/runs/R01/LEARNED.md) |
| R02 | Top-8 để đo recall–cost | 94.61111 official_score_0_100; Kaggle 94.5555/94.6666 Competition score (public/private) | [Đọc](inverse-translation/runs/R02/LEARNED.md) |
| R03 | Top-16 để đo recall–cost | 96.85897 official_score_0_100; Kaggle 100.0000/93.7179 Competition score (public/private) | [Đọc](inverse-translation/runs/R03/LEARNED.md) |
| R04 | Cross-entropy thay margin, cùng candidate budget | 100.00000 official_score_0_100; Kaggle 100.0000/100.0000 Competition score (public/private) | [Đọc](inverse-translation/runs/R04/LEARNED.md) |
| R05 | Coarse-to-fine top-8→32 | 100.00000 official_score_0_100; Kaggle 100.0000/100.0000 Competition score (public/private) | [Đọc](inverse-translation/runs/R05/LEARNED.md) |
| R06 | Coordinate sweep thay full replacement grid | 100.00000 official_score_0_100; Kaggle 100.0000/100.0000 Competition score (public/private) | [Đọc](inverse-translation/runs/R06/LEARNED.md) |
| R07 | Restart chỉ cho case chưa exact, đo tổng chi phí | 100.00000 official_score_0_100; Kaggle 100.0000/100.0000 Competition score (public/private) | [Đọc](inverse-translation/runs/R07/LEARNED.md) |
| R08 | Top8 exact search with contextual HotFlip fallback | 97.27778 official_score_0_100; Kaggle 94.5555/100.0000 Competition score (public/private) | [Đọc](inverse-translation/runs/R08/LEARNED.md) |
| R09 | Contextual gradient coordinate search from top1 probe seeds | 100.00000 official_score_0_100; Kaggle 100.0000/100.0000 Competition score (public/private) | [Đọc](inverse-translation/runs/R09/LEARNED.md) |

## Alchemy

| Run | Ý tưởng | Kết quả | Nhật ký |
|---|---|---|---|
| R01 | Cross-encoder + symmetry consistency | 0.10000 bijective_holdout_accuracy; Kaggle 0.0285/0.0000 Accuracy (public/private) | [Đọc](alchemy/runs/R01/LEARNED.md) |
| R02 | Hard negative mining từ train | 0.13333 bijective_holdout_accuracy; Kaggle 0.0000/0.0000 Accuracy (public/private) | [Đọc](alchemy/runs/R02/LEARNED.md) |
| R03 | Candidate-list CE thay sampled negatives | 0.10000 bijective_holdout_accuracy; Kaggle 0.0285/0.0285 Accuracy (public/private) | [Đọc](alchemy/runs/R03/LEARNED.md) |
| R04 | Bốn encoder layers từ đầu thay vì hai | 0.06667 bijective_holdout_accuracy; Kaggle 0.0000/0.0000 Accuracy (public/private) | [Đọc](alchemy/runs/R04/LEARNED.md) |
| R05 | Trung bình xác suất hai thứ tự thay trung bình logits | 0.06667 bijective_holdout_accuracy; Kaggle 0.0285/0.0000 Accuracy (public/private) | [Đọc](alchemy/runs/R05/LEARNED.md) |
| R06 | Sinkhorn loss để học cạnh tranh toàn cục trước Hungarian | 0.10000 bijective_holdout_accuracy; Kaggle 0.0000/0.0000 Accuracy (public/private) | [Đọc](alchemy/runs/R06/LEARNED.md) |
| R07 | Kết hợp hard negatives + full-list CE + probability averaging | 0.06667 bijective_holdout_accuracy; Kaggle 0.0285/0.0000 Accuracy (public/private) | [Đọc](alchemy/runs/R07/LEARNED.md) |
| R08 | Composer a+b / |a-b| / a*b có tính giao hoán sẵn | 0.10000 bijective_holdout_accuracy; Kaggle 0.0000/0.0000 Accuracy (public/private) | [Đọc](alchemy/runs/R08/LEARNED.md) |
| R09 | Tied input/result role space + residual additive composition | 0.16667 bijective_holdout_accuracy; Kaggle 0.0285/0.1142 Accuracy (public/private) | [Đọc](alchemy/runs/R09/LEARNED.md) |
| R10 | Square bijective minibatches with row/column CE and balanced Sinkhorn | 0.16667 bijective_holdout_accuracy; Kaggle 0.0285/0.1142 Accuracy (public/private) | [Đọc](alchemy/runs/R10/LEARNED.md) |
| R11 | ESZSL-inspired bilinear BPE and character attribute compatibility | 0.03333 bijective_holdout_accuracy; Kaggle 0.0000/0.0000 Accuracy (public/private) | [Đọc](alchemy/runs/R11/LEARNED.md) |

## Tally

| Run | Ý tưởng | Kết quả | Nhật ký |
|---|---|---|---|
| R01 | Scratch byte BPE word pooling + packed BiGRU, gold CE and expected entity-run counts | 0.44444 entity_span_micro_F1; Kaggle 0.2971/0.3008 Entity-span micro F1 (public/private) | [Đọc](tally/runs/R01/LEARNED.md) |
| R02 | CRF exact pair-marginal expected run counts | 0.44444 entity_span_micro_F1; Kaggle 0.3039/0.3035 Entity-span micro F1 (public/private) | [Đọc](tally/runs/R02/LEARNED.md) |
| R03 | CRF per-type count composite likelihood | 0.44444 entity_span_micro_F1; Kaggle 0.3530/0.3530 Entity-span micro F1 (public/private) | [Đọc](tally/runs/R03/LEARNED.md) |
| R04 | Byte character CNN supplements BPE count CRF | running | Chưa có kết quả |
| R05 | Approximate joint moment CRF teacher and stop-gradient KL | running | Chưa có kết quả |
