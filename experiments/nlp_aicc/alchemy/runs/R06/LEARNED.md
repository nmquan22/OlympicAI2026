# R06 · Sinkhorn training models candidate competition before Hungarian

Sinkhorn auxiliary loss đạt **3/30**, ngang R01. Mục tiêu cạnh tranh candidate lúc train hợp lý về cấu trúc nhưng không tự bổ sung semantic knowledge. Independent 2/30, greedy 2/30; Hungarian 3/30. Pool train hình chữ nhật và batch có thể trùng target; loss chỉ kích hoạt khi target khác nhau.

Mỗi lượt khởi tạo độc lập model/optimizer với seed 42; số thí nghiệm khác số Saved Version. Toàn batch đã hoàn tất; output riêng nằm trong alchemy/Rxx/.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
