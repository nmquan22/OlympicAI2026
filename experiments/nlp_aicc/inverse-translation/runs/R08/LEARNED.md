# Inverse Translation R08

R08 chỉ exact23/24 và official97.2778; Kaggle public94.5555/private100. Gradient fallback cải thiện so top8 cũ22/24 nhưng chưa giữ100. Code15.16s còn chậm hơn R06 nên không chọn làm bản tiết kiệm. Row21 còn margin0.01132; first-order CE proposal và shortlist hữu hạn không bảo đảm tìm được nghiệm exact.

Canary bốn provided examples khớp ở mỗi run (8/8 checks trong batch), BATCH_RESULT có đủR08/R09. Peak 331.45MiB trên T4. Không sửa frozen model weights.

[5 paper và thiết kế](../../../research_v2/inverse-translation.md).

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
