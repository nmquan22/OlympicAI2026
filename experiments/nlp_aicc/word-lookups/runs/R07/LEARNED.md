# R07 · Combine bidirectional surprise, cohesion, DP and punctuation

Kết hợp hai chiều, sentence normalization, cohesion PMI, DP và dấu câu. NLL giống R02 **5.8020** vì LM được fit cùng điều kiện; khác biệt segmentation đến từ decoder. Không giả định nhiều trick cộng lại sẽ tốt hơn. Tất cả threshold/statistics fit từ training fold, chưa đọc gold test.

Một Saved Version chứa các lượt độc lập; báo cáo count đúng mỗi config một lần. Batch mất1000.3s gồm startup và một R01 bị gọi lặp do cell cũ; R01 lặp không được tính thêm. Mọi run output riêng trong word-lookups/Rxx/.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.

## Boundary F1 thật sau khi đóng băng output

Kaggle public **0.9097**, private **0.9113**; UI hiển thị bốn chữ số. Đây là late submission luyện tập, không phải LM validation. Không đọc gold test, không chỉnh threshold sau khi thấy các điểm này.
