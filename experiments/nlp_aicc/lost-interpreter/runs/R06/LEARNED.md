# R06 · Program regression

Signed-log target đạt **MAE38.03**, không vượt R01 **36.26**. Train loss theo thang nén không trùng mục tiêu MAE raw; giảm tác động outlier lúc train có thể làm sai phân bổ lỗi. Đây là bản sạch version7; số38.16 của attempt trước được thay bằng kết quả này, không tính thành lượt mới.

Một seed, CUDA FP16 không ép mọi kernel deterministic. Kết quả retry có thể dao động nhẹ; mỗi config chỉ tính một lần.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
