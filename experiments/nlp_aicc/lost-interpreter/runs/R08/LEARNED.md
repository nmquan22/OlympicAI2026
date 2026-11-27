# R08 · Program regression

Ứng viên **BiGRU + packed sequences + masked mean/max + SmoothL1 beta0.2** đạt **MAE15.31**, giảm **57.8%** so R01 **36.26** trên cùng3,000 holdout. Code146.0s, peak121MiB. Đây là thay cả backbone/loss, không đủ để gán toàn bộ hiệu ứng cho GRU. So reference MAE22.65 chia fold khác không phải đối chứng cặp.

Flow: BPE train-only → embedding128 → packed two-layer BiGRU128 mỗi chiều → mean/max có mask → MLP scalar. SmoothL1 trên target chuẩn hóa. Không pretrained.

Lỗi theo số lệnh: ≤4 lệnh **MAE5.91 (757)**;5–8 lệnh **16.64 (1475)**;9–16 lệnh **22.01 (768)**. Model tốt hơn nhiều nhưng chưa có bằng chứng thực thi chính xác chương trình dài. Hướng tiếp là trạng thái/giá trị theo từng instruction, không chỉ thêm attention.

Ví dụ train thật có OP5 -5, OP2 1, OP11 7 10; các opcode có số trong tên, nên parser numeric phải tránh đọc nhầm chữ số opcode thành operand. Chưa đoán thủ công semantic opcode từ vài mẫu rồi gán nhãn test.

Một seed, CUDA FP16 không ép mọi kernel deterministic. Kết quả retry có thể dao động nhẹ; mỗi config chỉ tính một lần.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
