# R06 · Explicit punctuation boundaries enforce consistent BMES

Thêm boundary trước/sau dấu câu Unicode rồi xuất BMES hợp lệ. Can thiệp nhỏ, có cơ chế rõ; có thể cải thiện format/ngữ pháp nhưng không nhất thiết tăng F1. NLL giữ6.1068. Punctuation-only là ablation decoder, không phải model mới.

Một Saved Version chứa các lượt độc lập; báo cáo count đúng mỗi config một lần. Batch mất1000.3s gồm startup và một R01 bị gọi lặp do cell cũ; R01 lặp không được tính thêm. Mọi run output riêng trong word-lookups/Rxx/.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.

## Boundary F1 thật sau khi đóng băng output

Kaggle public **0.8196**, private **0.8215**; UI hiển thị bốn chữ số. Đây là late submission luyện tập, không phải LM validation. Không đọc gold test, không chỉnh threshold sau khi thấy các điểm này.
