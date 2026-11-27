# R07 · Selective restarts spend compute only on non-exact rows

Selective restart giữ100 và có đúng cùng **10,376** candidate evaluations nhưR05: cả24 câu đã exact nên restart không được kích hoạt. Chưa có bằng chứng restart giúp bộ target này; không gán công cho một nhánh không chạy. Cần case khó hơn để đánh giá robustness, không cần thêm seed chỉ để đủ lượt.

Đây là input optimization trên target công khai do đề cấp, không phải accuracy tổng quát hóa trên test ẩn. Reference cũng đạt100. Mỗi experiment load model được cấp độc lập, giữ token IDs gốc; không pretrained mới và không train thêm weights.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
