# R04 · Cấu trúc chương trình

Learned query pooling thay mean pooling: MAE **37.77**, vẫn kém R01 **36.26**. So với R02 37.94 chỉ cải thiện nhỏ. Một query toàn cục vẫn chưa giữ được trạng thái thực thi từng bước.

Một seed, holdout cố định 15%; điểm reference chia validation khác nên không phải so sánh cặp.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
