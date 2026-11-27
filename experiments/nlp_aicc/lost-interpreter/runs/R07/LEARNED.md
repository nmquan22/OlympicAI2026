# R07 · Program regression

Kết hợp marker+numeric branch+query pooling+auxiliary đạt **MAE37.58**, vẫn kém R01 **36.26**. Nhiều thành phần không cộng điểm tự động; global statistics không mô phỏng thứ tự opcode. Không dùng cấu hình phức tạp nhất làm default. Bản sạch version7, attempt trước không được tính thêm.

Một seed, CUDA FP16 không ép mọi kernel deterministic. Kết quả retry có thể dao động nhẹ; mỗi config chỉ tính một lần.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.

## Hạn chế parser đã phát hiện khi đọc lỗi

Source SHA của lượt này dùng regex có thể đọc nhầm chữ số cuối OP11/OP12 thành operand. Vì vậy kết quả numeric branch không là bằng chứng sạch rằng numerical features vô ích. Code hiện tại đã sửa lookbehind và kiểm tra trên ví dụ train; số đã đo vẫn gắn với source SHA cũ, không đổi metric lịch sử. Ứng viên tốt nhất R08 không dùng numeric branch. Chưa chạy một ablation GPU riêng cho parser sửa.
