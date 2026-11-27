# Tiến độ thực nghiệm NLP

**29 thí nghiệm T4 thành công đã lưu bằng chứng; mục tiêu tối thiểu 35 (7/bài).** Số thí nghiệm khác số Saved Version; lượt lỗi không được tính. Tối đa hai Saved Version đang chạy.

| Bài | Đã hoàn tất | Tốt nhất trong điều kiện đã chạy |
|---|---:|---|
| [Moment Match](moment-match/IDEAS.md) | 7/7 | R04: 0.69122 accuracy |
| [Lost Interpreter](lost-interpreter/IDEAS.md) | 5/7 | R01: 36.26026 MAE |
| [Word Lookups](word-lookups/IDEAS.md) | 3/7 | Chưa xếp hạng segmentation: cần Boundary F1 |
| [Inverse Translation](inverse-translation/IDEAS.md) | 7/7 | R01: 100.00000 official_score_0_100 |
| [Alchemy](alchemy/IDEAS.md) | 7/7 | R02: 0.13333 bijective_holdout_accuracy |

Accuracy/MAE trên validation, LM NLL proxy và official inverse score là các metric khác nhau. Không so điểm ngang giữa các bài; không gọi NLL là Boundary F1.

## Moment Match

| Run | Ý tưởng | Kết quả | Nhật ký |
|---|---|---|---|
| R01 | BPE + soft late interaction | 0.65439 accuracy | [Đọc](moment-match/runs/R01/LEARNED.md) |
| R02 | Thêm Transformer ở mức token | 0.66383 accuracy | [Đọc](moment-match/runs/R02/LEARNED.md) |
| R03 | Thêm positional Transformer ở mức sự kiện | 0.60434 accuracy | [Đọc](moment-match/runs/R03/LEARNED.md) |
| R04 | IDF weighting cho alignment | 0.69122 accuracy | [Đọc](moment-match/runs/R04/LEARNED.md) |
| R05 | Giảm nhiệt độ soft alignment | 0.66100 accuracy | [Đọc](moment-match/runs/R05/LEARNED.md) |
| R06 | Chỉ giữ sự kiện trung tâm để đo vai trò ngữ cảnh | 0.49103 accuracy | [Đọc](moment-match/runs/R06/LEARNED.md) |
| R07 | Kết hợp token attention + IDF, dựa trên R03/R04 | 0.66478 accuracy | [Đọc](moment-match/runs/R07/LEARNED.md) |
| R08 | IDF + prior căn chỉnh đơn điệu khả vi | planned | Chưa có kết quả |
| R09 | IDF + max alignment gần reference | planned | Chưa có kết quả |

## Lost Interpreter

| Run | Ý tưởng | Kết quả | Nhật ký |
|---|---|---|---|
| R01 | BPE Transformer + mean/max + MAE | 36.26026 MAE | [Đọc](lost-interpreter/runs/R01/LEARNED.md) |
| R02 | Thêm separator lệnh và marker số | 37.94136 MAE | [Đọc](lost-interpreter/runs/R02/LEARNED.md) |
| R03 | Nhánh thống kê số fit trên train | 37.66859 MAE | [Đọc](lost-interpreter/runs/R03/LEARNED.md) |
| R04 | Learned query attention pooling | 37.77339 MAE | [Đọc](lost-interpreter/runs/R04/LEARNED.md) |
| R05 | Auxiliary loss dự đoán số lệnh | 37.60918 MAE | [Đọc](lost-interpreter/runs/R05/LEARNED.md) |
| R06 | Signed-log target để kiểm tra outlier | planned | Chưa có kết quả |
| R07 | Kết hợp structure + numeric + query + auxiliary | planned | Chưa có kết quả |
| R08 | BPE BiGRU + packed masking + SmoothL1 theo cùng holdout | planned | Chưa có kết quả |

## Word Lookups

| Run | Ý tưởng | Kết quả | Nhật ký |
|---|---|---|---|
| R01 | Transformer surprisal với offset-safe BPE | 6.10684 BPE_LM_NLL_proxy | [Đọc](word-lookups/runs/R01/LEARNED.md) |
| R02 | Surprisal hai chiều | 5.80200 BPE_LM_NLL_proxy | [Đọc](word-lookups/runs/R02/LEARNED.md) |
| R03 | Chuẩn hóa surprisal theo vị trí/độ dài | 6.10684 BPE_LM_NLL_proxy | [Đọc](word-lookups/runs/R03/LEARNED.md) |
| R04 | Kết hợp cohesion n-gram học từ train | planned | Chưa có kết quả |
| R05 | DP với prior độ dài từ | planned | Chưa có kết quả |
| R06 | Punctuation-aware BMES constraints | planned | Chưa có kết quả |
| R07 | Kết hợp hai chiều + cohesion + DP | planned | Chưa có kết quả |

## Inverse Translation

| Run | Ý tưởng | Kết quả | Nhật ký |
|---|---|---|---|
| R01 | Probe streaming giữ top-32, margin search | 100.00000 official_score_0_100 | [Đọc](inverse-translation/runs/R01/LEARNED.md) |
| R02 | Top-8 để đo recall–cost | 94.61111 official_score_0_100 | [Đọc](inverse-translation/runs/R02/LEARNED.md) |
| R03 | Top-16 để đo recall–cost | 96.85897 official_score_0_100 | [Đọc](inverse-translation/runs/R03/LEARNED.md) |
| R04 | Cross-entropy thay margin, cùng candidate budget | 100.00000 official_score_0_100 | [Đọc](inverse-translation/runs/R04/LEARNED.md) |
| R05 | Coarse-to-fine top-8→32 | 100.00000 official_score_0_100 | [Đọc](inverse-translation/runs/R05/LEARNED.md) |
| R06 | Coordinate sweep thay full replacement grid | 100.00000 official_score_0_100 | [Đọc](inverse-translation/runs/R06/LEARNED.md) |
| R07 | Restart chỉ cho case chưa exact, đo tổng chi phí | 100.00000 official_score_0_100 | [Đọc](inverse-translation/runs/R07/LEARNED.md) |

## Alchemy

| Run | Ý tưởng | Kết quả | Nhật ký |
|---|---|---|---|
| R01 | Cross-encoder + symmetry consistency | 0.10000 bijective_holdout_accuracy | [Đọc](alchemy/runs/R01/LEARNED.md) |
| R02 | Hard negative mining từ train | 0.13333 bijective_holdout_accuracy | [Đọc](alchemy/runs/R02/LEARNED.md) |
| R03 | Candidate-list CE thay sampled negatives | 0.10000 bijective_holdout_accuracy | [Đọc](alchemy/runs/R03/LEARNED.md) |
| R04 | Bốn encoder layers từ đầu thay vì hai | 0.06667 bijective_holdout_accuracy | [Đọc](alchemy/runs/R04/LEARNED.md) |
| R05 | Trung bình xác suất hai thứ tự thay trung bình logits | 0.06667 bijective_holdout_accuracy | [Đọc](alchemy/runs/R05/LEARNED.md) |
| R06 | Sinkhorn loss để học cạnh tranh toàn cục trước Hungarian | 0.10000 bijective_holdout_accuracy | [Đọc](alchemy/runs/R06/LEARNED.md) |
| R07 | Kết hợp hard negatives + full-list CE + probability averaging | 0.06667 bijective_holdout_accuracy | [Đọc](alchemy/runs/R07/LEARNED.md) |
| R08 | Composer a+b / |a-b| / a*b có tính giao hoán sẵn | planned | Chưa có kết quả |
