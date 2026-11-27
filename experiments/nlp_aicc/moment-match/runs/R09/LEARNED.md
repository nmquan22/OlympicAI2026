# R09 · Hard max word-piece alignment instead of soft matching

Hard max alignment đạt **67.71%**, kém R04 soft alignment+IDF **69.12%**. Max gần reference không bảo đảm hiệu quả khi word đã bị tách thành BPE piece rồi khởi tạo bằng trung bình nhiều word vector. Đây là giả thuyết mất thông tin ở feature, chưa phải bằng chứng xác nhận nguyên nhân.

R10 sẽ giữ BPE nhưng thêm một view từ nguyên word vectors được đề cấp, tránh trung bình nhiều nghĩa vào cùng subword. Không tải checkpoint mới.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
