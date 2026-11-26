# R01 · Đã học được gì

BPE + soft late interaction đạt accuracy **0.65439** trên 1.059 mẫu validation chính thức. Best ở epoch 3; train loss tiếp tục giảm tới epoch 10 nhưng validation không tăng, cho thấy bắt đầu overfit. Chương trình chạy 59.12s, peak allocated VRAM 211.65 MiB trên Tesla T4.

Nguồn reference ghi 0.7243, cao hơn run này khoảng 6.99 điểm phần trăm. Chưa phải so sánh có kiểm soát vì tokenization, vector khởi tạo, pooling, chiều hidden và optimizer khác nhau; chưa thể gán toàn bộ chênh lệch cho BPE. Hypothesis tiếp theo: contextual token/event attention và IDF có lấy lại thông tin bị mean-initialization làm mờ không?

R01 đã xuất 1.970 prediction hợp lệ. Chưa nộp leaderboard; không biến accuracy validation thành điểm Kaggle.

Chưa trích lỗi theo từng mẫu cho R01; nhận xét hiện tại chỉ dựa trên đường train/validation.
