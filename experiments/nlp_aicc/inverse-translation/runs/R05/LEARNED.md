# R05 · Coarse-to-fine proposal search top-8 then top-32

Coarse top8 rồi mở top32 giữ100, **10,376** candidate evaluations và13.47s. Không giảm số candidate soR01, dù wall time thấp hơn. Warm-up/allocator và cách chia batch có thể ảnh hưởng runtime; không gọi đây là lợi ích bền vững nếu chỉ nhìn một lượt.

Đây là input optimization trên target công khai do đề cấp, không phải accuracy tổng quát hóa trên test ẩn. Reference cũng đạt100. Mỗi experiment load model được cấp độc lập, giữ token IDs gốc; không pretrained mới và không train thêm weights.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
