# R07 · Program regression

Kết hợp marker+numeric branch+query pooling+auxiliary đạt **MAE37.58**, vẫn kém R01 **36.26**. Nhiều thành phần không cộng điểm tự động; global statistics không mô phỏng thứ tự opcode. Không dùng cấu hình phức tạp nhất làm default. Bản sạch version7, attempt trước không được tính thêm.

Một seed, CUDA FP16 không ép mọi kernel deterministic. Kết quả retry có thể dao động nhẹ; mỗi config chỉ tính một lần.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
