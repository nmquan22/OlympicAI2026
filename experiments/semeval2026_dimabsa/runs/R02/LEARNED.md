# R02 · Transformer pre-norm scratch: bỏ cấu hình hiện tại

**Dev RMSE_VA 1.62459, kém Ridge 1.45154.** Lượt GPU đã hoàn tất; không phải điểm giả lập. [Kaggle Version 1](https://www.kaggle.com/code/quannguyen0501/dimabsa-r02-scratch-prenorm-transformer?scriptVersionId=374890664), [report](report.json), [log epoch](epochs.jsonl), [scorer BTC](official_scorer.log), [phân tích lỗi](analysis.json).

## Model đã chạy

Text + aspect → BPE fit train sạch → embedding random 192 → 4 tầng Transformer pre-norm, 4 heads → mean/max text, aspect pool và attention dùng aspect làm query → MLP → `1 + 8*sigmoid` cho VA. 2.534.980 tham số; không weights/tokenizer pretrained. AdamW, LR 0.0005, warmup 10%, cosine, MSE; seed 42.

Train 2.336 cặp text–aspect sau khi loại một câu trùng dev. Dev giữ nguyên 200 câu/340 aspects. Best epoch 15; dừng epoch 25 vì 10 epoch không cải thiện. Model + preprocessing mất 40.16 giây, toàn Saved Version 75.4 giây theo viewer; không cộng thời gian queue vào thời gian train. Checkpoint/tokenizer giữ trong output Kaggle, không commit weights vào repo.

## Thua ở đâu?

| Nhóm dev | Số aspect | Ridge RMSE_VA ↓ | Transformer RMSE_VA ↓ |
|---|---:|---:|---:|
| V < 5 | 56 | 2.51456 | 3.13896 |
| V ≥ 5 | 284 | 1.12944 | 1.10311 |
| A < 6 | 49 | 2.08631 | 2.86302 |
| Câu > 30 từ | 11 | 2.59605 | 3.71777 |

Ví dụ: `The only downside , was that the service was really slow , and everything is extremely over priced`, aspect `everything`: gold **1.33#8.83**, Transformer **7.46#7.57**, Ridge **5.20#6.71**. Model dự đoán sai hẳn valence dù có từ/cụm nêu hạn chế trong câu.

Loss train giảm từ 2.185 ở epoch 1 xuống 0.261 ở epoch 25, trong khi dev tốt nhất ở epoch 15 rồi dao động xấu hơn: bằng chứng overfit trong cấu hình này. Giả thuyết cho lượt kế tiếp phải xử lý biểu diễn opinion/khái quát sentiment âm, thay vì chỉ tăng số tầng hoặc thử nhiều dropout.

Bootstrap 2.000 lần theo **câu** (giữ các aspect của cùng câu trong một cluster): chênh RMSE Transformer − Ridge = **+0.17305**, khoảng 95% **[+0.00365, +0.35329]**. Đây là độ bất định trên dev và hai prediction đã cố định; không đo biến thiên qua nhiều seed hay hiệu chỉnh thiên lệch do chọn epoch trên dev.

## Quyết định

Loại **cấu hình R02 hiện tại** khỏi nhánh ưu tiên và không cấp thêm lượt chỉ để tăng độ sâu/chỉnh dropout. Chờ R01 BiLSTM attention trước khi chọn backbone cho nâng cấp. Không kết luận mọi Transformer scratch đều yếu hơn LSTM từ một lượt này.

Bài học phòng thi: thêm kiến trúc hiện đại chỉ có ích khi nó xử lý nhóm lỗi đang làm mất điểm. Với tập train nhỏ, kiểm tra polarity theo nhóm và log train–dev trước khi dùng các lượt còn lại vào tuning.
