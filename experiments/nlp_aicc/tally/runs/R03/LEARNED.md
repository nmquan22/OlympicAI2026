# Tally R03

Weighted per-type count likelihood tăng public .29710→.35301, private .30080→.35300 (private chỉ ghi, không dùng để chọn). ValF1 vẫn6TP/6FP/9FN trên10 câu: pilot quá nhỏ để phân biệt. Weak exact count vector31.5%→36.125%, MAE .22375→.21975. ORG dự đoán3 thay1 nhưng vẫn đúng0/5 vì hụt biên; lỗi biên cùng loại tăng3→5. Mất dự đoán PERSON giả ở từ director nhưng chưa tìm được CREATIVE_WORK. Đúng count hơn chưa đủ đúng span.

T4 code374.102028s, peak136.327637MiB; Kaggle tổng405.2s. Canary partition/count expectation/count numerator/Viterbi/padding/finite gradients đều pass trước train; BATCH_RESULT đầy đủ.

Giả thuyết tiếp: character CNN hỗ trợ tên hiếm (14/15 gold entity chứa từ chưa gặp), và PR teacher điều chỉnh posterior theo năm count đồng thời. PR phải log residual thực, không gọi moment matching hoặc teacher chưa hội tụ là projection chính xác. [Năm paper](../../RESEARCH.md).

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
