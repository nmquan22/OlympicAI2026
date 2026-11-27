# R05 · Dynamic programming adds a word-length prior

DP chọn segmentation toàn câu với xác suất boundary và prior độ dài, max word length6. Khác threshold độc lập: có cạnh tranh giữa các cách cắt. Prior lấy từ độ dài BPE piece train không phải độ dài từ gold; có thể áp bias sai. NLL giữ6.1068.

Một Saved Version chứa các lượt độc lập; báo cáo count đúng mỗi config một lần. Batch mất1000.3s gồm startup và một R01 bị gọi lặp do cell cũ; R01 lặp không được tính thêm. Mọi run output riêng trong word-lookups/Rxx/.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
