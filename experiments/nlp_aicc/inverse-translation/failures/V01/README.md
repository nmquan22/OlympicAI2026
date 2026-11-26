# V01 không tính vào 7 lượt

Model được cấp tải thành công trên T4 nhưng assertion so sánh `generate` với ví dụ gốc thất bại. Chưa chạy search, chưa tạo submission. Không gỡ assertion để lấy điểm.

Môi trường Kaggle hiện dùng Transformers 5 (log loader mới). Reference chạy với API FSMT cũ; kiểm tra phiên bản tương thích 4.44.2 và giữ nguyên bốn ví dụ làm canary. Đây là giả thuyết sửa tương thích, chưa xác nhận cho tới khi canary pass.
