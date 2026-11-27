# R08 · IDF plus a differentiable monotonic subsequence alignment prior

DP canary đối chiếu **10 đường vét cạn**, gradient hữu hạn, passed. Accuracy **68.93%**, chưa vượt R04 **69.12%** (kém hai câu). Prior thứ tự có cơ chế đúng nhưng có thể trùng phần CNN đã học hoặc quá cứng với mô tả sự kiện bị lược. Một seed không đủ kết luận hai model khác nhau ổn định.

R10 sẽ giữ BPE nhưng thêm một view từ nguyên word vectors được đề cấp, tránh trung bình nhiều nghĩa vào cùng subword. Không tải checkpoint mới.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
