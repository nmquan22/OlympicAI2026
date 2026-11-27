# R02 · Bidirectional surprisal sees boundaries from both contexts

Huấn luyện causal LM với cả văn bản thuận và đảo, rồi gộp surprisal hai phía. NLL5.8020 thấp hơn số R01 nhưng BPE/corpus đã đổi nên không dùng chênh NLL làm bằng chứng tách từ tốt. Boundary F1 public đã quan sát **0.8179**, private **0.8214**, so R01 **0.7559/0.7560**. Đây là ứng viên kết hợp corpus/vocab/direction, chưa tách riêng hiệu ứng của từng yếu tố.

Một Saved Version chứa các lượt độc lập; báo cáo count đúng mỗi config một lần. Batch mất1000.3s gồm startup và một R01 bị gọi lặp do cell cũ; R01 lặp không được tính thêm. Mọi run output riêng trong word-lookups/Rxx/.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
