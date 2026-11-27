# R07 · Combine hard negatives, full-list CE and probability averaging

Full-list CE + probability averaging đạt **2/30**, kém hard negatives riêng R02 **4/30**. Khi đã dùng toàn bộ candidate, hard-negative mining không đổi tập negative nữa; không gán tăng/giảm cho mining trong cấu hình này. Đây thực chất là kiểm tra full-list CE kết hợp phép gộp xác suất. Không chọn bản phức tạp nhất chỉ vì nhiều trick.

Mỗi lượt khởi tạo độc lập model/optimizer với seed 42; số thí nghiệm khác số Saved Version. Toàn batch đã hoàn tất; output riêng nằm trong alchemy/Rxx/.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
