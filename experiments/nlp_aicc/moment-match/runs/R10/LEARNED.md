# R10 · Giữ nguyên word feature chưa đủ để tăng điểm

BPE view + nguyên word-vector view được đề cấp đạt **64.87%**, không vượt IDF R04 **69.12%**. Canary finite cho cả input bình thường và word view toàn padding đã passed; IDs shift+1 đúng reference. Notebook thành công96.7s, phần code67.4s.

Điều này bác bỏ ứng viên ghép bốn channel bằng CNN trong ngân sách hiện tại, không chứng minh word vectors vô ích. Fusion đổi cả feature và số channel/khởi tạo CNN; chưa cô lập nguyên nhân giảm điểm. Giữ R04 làm lựa chọn hiện tại.

Sau bảy lượt ban đầu đã thử thêm ba giả thuyết: monotonic DP68.93%, hard BPE max67.71%, dual feature64.87%. Không chọn model mới chỉ vì có thêm cơ chế. BPE vẫn là tokenizer chính; nhánh phụ chỉ dùng tài nguyên fastText của đề, không checkpoint mới.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
