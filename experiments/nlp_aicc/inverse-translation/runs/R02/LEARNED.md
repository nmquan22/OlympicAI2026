# R02 · Top-8 proposal set trades candidate recall for speed

Top8 còn **94.61**, exact **22/24**; chỉ2640 candidate evaluations so9624 ở top32. Proposal nhỏ nhanh hơn nhưng có thể loại mất Alien token cần thiết, search downstream không cứu được candidate đã bị cắt. Không chọn top8 nếu mục tiêu điểm tuyệt đối.

Đây là input optimization trên target công khai do đề cấp, không phải accuracy tổng quát hóa trên test ẩn. Reference cũng đạt100. Mỗi experiment load model được cấp độc lập, giữ token IDs gốc; không pretrained mới và không train thêm weights.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
