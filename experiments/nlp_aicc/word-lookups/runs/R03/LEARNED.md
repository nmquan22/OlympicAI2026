# R03 · Sentence-level normalization tests surprisal scale shift

Giữ đúng LM R01, chuẩn hóa surprisal trong từng câu trước đặt ngưỡng train. NLL không đổi **6.1068** là đúng kỳ vọng vì chỉ decoder đổi. Giả thuyết: giảm lệch thang điểm giữa câu dễ và câu khó. Boundary F1 được chấm riêng sau khi đóng băng output.

Một Saved Version chứa các lượt độc lập; báo cáo count đúng mỗi config một lần. Batch mất1000.3s gồm startup và một R01 bị gọi lặp do cell cũ; R01 lặp không được tính thêm. Mọi run output riêng trong word-lookups/Rxx/.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.

## Boundary F1 thật sau khi đóng băng output

Kaggle public **0.7604**, private **0.7594**; UI hiển thị bốn chữ số. Đây là late submission luyện tập, không phải LM validation. Không đọc gold test, không chỉnh threshold sau khi thấy các điểm này.
