# R03 · CE over the complete training candidate list

CE toàn bộ training candidate đạt **3/30**, ngang R01. Loại bỏ ngẫu nhiên negative sampling chưa tạo được kiến thức cho result name mới. Independent 2/30, greedy 3/30. Chi phí cao hơn sampled CE nhưng chưa tốt hơn trên holdout này.

Mỗi lượt khởi tạo độc lập model/optimizer với seed 42; số thí nghiệm khác số Saved Version. Toàn batch đã hoàn tất; output riêng nằm trong alchemy/Rxx/.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
