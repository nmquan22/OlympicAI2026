# R04 · Four scratch encoder layers test extra capacity

Bốn scratch Transformer layer đạt **2/30**, thấp hơn hai layer R01 **3/30**. Dữ liệu ít không bảo đảm hưởng lợi từ tăng capacity. Dùng cùng split và seed; không áp dụng pretrained để che khác biệt.

Mỗi lượt khởi tạo độc lập model/optimizer với seed 42; số thí nghiệm khác số Saved Version. Toàn batch đã hoàn tất; output riêng nằm trong alchemy/Rxx/.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
