# R04 · Train-only character cohesion supplements BPE-internal boundaries

Kết hợp surprisal với PMI giữa ký tự fit hoàn toàn từ train. PMI cao là dấu hiệu hai ký tự gắn với nhau; dùng -PMI để giảm xu hướng cắt ở đó. Không từ điển bên ngoài, không gold test. NLL giữ6.1068; chỉ Boundary F1 mới kết luận decoding có ích.

Một Saved Version chứa các lượt độc lập; báo cáo count đúng mỗi config một lần. Batch mất1000.3s gồm startup và một R01 bị gọi lặp do cell cũ; R01 lặp không được tính thêm. Mọi run output riêng trong word-lookups/Rxx/.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.

## Boundary F1 thật sau khi đóng băng output

Kaggle public **0.8766**, private **0.8775**; UI hiển thị bốn chữ số. Đây là late submission luyện tập, không phải LM validation. Không đọc gold test, không chỉnh threshold sau khi thấy các điểm này.
