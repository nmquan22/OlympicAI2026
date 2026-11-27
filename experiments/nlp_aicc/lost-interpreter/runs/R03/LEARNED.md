# R03 · Thống kê số chỉ cứu một phần

MAE **37.67**, tốt hơn R02 37.94 một ít nhưng vẫn kém R01 36.26. So với R02, thêm nhánh MLP 8 thống kê số (count, sum, mean, min, max, std, tổng trị tuyệt đối, số lệnh), signed-log và chuẩn hóa fit từ train fold.

Nhánh này giữ giá trị số mà BPE không biểu diễn tốt, nhưng gộp số thành thống kê làm mất **thứ tự lệnh, opcode và điều kiện nhảy**. Chưa đủ để mô phỏng một interpreter. Nếu chỉ chọn từ ba lượt hiện có vẫn chọn R01. Không khẳng định numeric features vô ích ở kiến trúc khác.

## Hạn chế parser đã phát hiện khi đọc lỗi

Source SHA của lượt này dùng regex có thể đọc nhầm chữ số cuối OP11/OP12 thành operand. Vì vậy kết quả numeric branch không là bằng chứng sạch rằng numerical features vô ích. Code hiện tại đã sửa lookbehind và kiểm tra trên ví dụ train; số đã đo vẫn gắn với source SHA cũ, không đổi metric lịch sử. Ứng viên tốt nhất R08 không dùng numeric branch. Chưa chạy một ablation GPU riêng cho parser sửa.
