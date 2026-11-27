# R05 · Average probabilities rather than logits of swapped inputs

Trung bình xác suất của hai thứ tự đạt **2/30**, thấp hơn logit averaging R01 **3/30**. Đây là phép gộp phi tuyến có thể đổi Hungarian assignment; temperature chung hoặc cộng hằng số mỗi hàng thì không. Chưa thấy lợi ích ở ứng viên hiện tại.

Mỗi lượt khởi tạo độc lập model/optimizer với seed 42; số thí nghiệm khác số Saved Version. Toàn batch đã hoàn tất; output riêng nằm trong alchemy/Rxx/.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
