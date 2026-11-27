# R02 · Model-mined hard negatives with a random-negative mixture

Hard negatives đạt **4/30**, hơn R01 **3/30** một câu. Model chọn nhầm candidate khó được đưa lại vào pool, trộn với random negatives để không chỉ nhìn một vùng hẹp. Trên cùng score matrix, independent **0/30**, greedy **1/30**, Hungarian **4/30**: candidate competition quan trọng. Chỉ 30 mẫu nên mức tăng một câu chưa đủ đáng tin.

Mỗi lượt khởi tạo độc lập model/optimizer với seed 42; số thí nghiệm khác số Saved Version. Toàn batch đã hoàn tất; output riêng nằm trong alchemy/Rxx/.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
