# R04 · Teacher-forced CE instead of decision margin

Cross-entropy cũng đạt100 nhưng **41,464** candidate evaluations, **34.10s**, so margin9624 và16.29s. Margin tối ưu điều kiện quyết định target đứng đầu; CE còn cố tăng confidence ngay khi chuỗi đã đúng. Early stop trong code cũng chỉ có zero-margin nên chi phí khác phản ánh cả objective lẫn điều kiện dừng.

Đây là input optimization trên target công khai do đề cấp, không phải accuracy tổng quát hóa trên test ẩn. Reference cũng đạt100. Mỗi experiment load model được cấp độc lập, giữ token IDs gốc; không pretrained mới và không train thêm weights.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
