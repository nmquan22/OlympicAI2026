# R03 · Thống kê số chỉ cứu một phần

MAE **37.67**, tốt hơn R02 37.94 một ít nhưng vẫn kém R01 36.26. So với R02, thêm nhánh MLP 8 thống kê số (count, sum, mean, min, max, std, tổng trị tuyệt đối, số lệnh), signed-log và chuẩn hóa fit từ train fold.

Nhánh này giữ giá trị số mà BPE không biểu diễn tốt, nhưng gộp số thành thống kê làm mất **thứ tự lệnh, opcode và điều kiện nhảy**. Chưa đủ để mô phỏng một interpreter. Nếu chỉ chọn từ ba lượt hiện có vẫn chọn R01. Không khẳng định numeric features vô ích ở kiến trúc khác.
