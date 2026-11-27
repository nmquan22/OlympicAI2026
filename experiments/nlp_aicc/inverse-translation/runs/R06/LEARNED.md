# R06 · Coordinate sweeps replace the full substitution grid

Coordinate sweep giữ **100**, giảm candidate evaluations **9624→7832 (18.6%)** và runtime **16.29→11.97s (26.5%)** trong batch. Model calls tăng182→268 do batch nhỏ hơn: số calls không đồng nghĩa tổng chi phí. Đây là ứng viên hiệu quả nhất giữ điểm tuyệt đối trên24 target hiện có; chưa kiểm định timing nhiều lần.

Đây là input optimization trên target công khai do đề cấp, không phải accuracy tổng quát hóa trên test ẩn. Reference cũng đạt100. Mỗi experiment load model được cấp độc lập, giữ token IDs gốc; không pretrained mới và không train thêm weights.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
