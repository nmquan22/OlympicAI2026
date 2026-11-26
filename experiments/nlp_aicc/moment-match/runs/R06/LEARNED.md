# R06 · Context có ích, cách ghép context mới là vấn đề

Chỉ giữ hàng sự kiện trung tâm trong alignment grid đạt **49.10%**, giảm **16.34 điểm phần trăm** so với R01 65.44%. Các điều kiện khác giữ nguyên; candidate window vẫn đầy đủ, target context bị che sau matching.

Không thể kết luận context vô ích từ việc R03 event Transformer giảm điểm. R06 cho thấy lân cận mang nhiều bằng chứng, còn R03 chỉ bác bỏ cách broadcast event context đang thử. Hướng hợp lý là giữ late interaction chi tiết và cải thiện trọng số token, thay vì loại lân cận.

Đây là ablation chẩn đoán cơ chế, không chọn làm mô hình nộp. Một seed, cùng 1,059 validation; không có leaderboard score trong báo cáo này.
