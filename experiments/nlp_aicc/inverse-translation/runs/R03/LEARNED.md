# R03 · Top-16 measures the middle recall-cost tradeoff

Top16 đạt **96.86**, exact **23/24**;5720 candidate evaluations. Đây là điểm trung gian giữa top8 và top32, vẫn chưa giữ100. Giá trị similarity của câu sai và exact rate khác nhau; không diễn giải96.86 là96.86% câu đúng.

Đây là input optimization trên target công khai do đề cấp, không phải accuracy tổng quát hóa trên test ẩn. Reference cũng đạt100. Mỗi experiment load model được cấp độc lập, giữ token IDs gốc; không pretrained mới và không train thêm weights.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
