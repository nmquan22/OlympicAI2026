# R01 · Surprisal là giả thuyết về ranh giới

LM NLL **6.1068** trên 3,709 train holdout; tạo đủ **9,272** chuỗi BMES đúng chiều dài. Kaggle total runtime **137.9s**, phần code **110.1s**. Không có Boundary F1 trong số này.

Flow: BPE train-only → causal Transformer từ đầu → surprisal ở vị trí đầu subword → ánh xạ offset về ký tự → ngưỡng fit trên train → BMES. BPE pieces không phải từ ngôn ngữ học. Tokenizer/decoding có thể hợp lệ nhưng segmentation vẫn thấp.

R02–R07 kiểm tra hướng ngược, chuẩn hóa, PMI, DP length prior và dấu câu. Cần điểm segmentation để kết luận cải thiện tách từ, không chọn NLL thay thế F1.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.

## Boundary F1 thật sau khi đóng băng output

Kaggle public **0.7559**, private **0.7560**; UI hiển thị bốn chữ số. Đây là late submission luyện tập, không phải LM validation. Không đọc gold test, không chỉnh threshold sau khi thấy các điểm này.
