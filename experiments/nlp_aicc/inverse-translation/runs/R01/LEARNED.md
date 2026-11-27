# R01 · Streaming top-32 probe avoids materializing the vocabulary product

Top32 + margin đạt **100, 24/24 exact**. Streaming probe giữ top-k qua từng batch, không tạo toàn ma trận từ vựng nguồn × đích. Bộ nhớ peak359MiB, probe3.82s; canary bốn ví dụ đều khớp. Cài Transformers4.57.1 với tokenizers0.22.2 tương thích Python3.13. Bản sạch hoàn tất, thay thế các retry lỗi trước.

Đây là input optimization trên target công khai do đề cấp, không phải accuracy tổng quát hóa trên test ẩn. Reference cũng đạt100. Mỗi experiment load model được cấp độc lập, giữ token IDs gốc; không pretrained mới và không train thêm weights.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
