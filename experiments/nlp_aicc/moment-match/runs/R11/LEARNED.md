# moment-match R11

Cross-attention event-level chưa tăng: val.6572, Kaggle.6304/.6233. Event pooling trước align có thể mất tín hiệu local token; attention thêm không tự tạo reasoning tốt hơn. Cần kiểm lỗi, không tăng depth mù.

[Paper và thiết kế](../../../research_v2/moment-match.md). Code 65.001s, peak 230.03MiB trên T4.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
