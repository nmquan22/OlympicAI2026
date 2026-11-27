# R09 · Chia sẻ không gian vai trò cho tên ít dữ liệu

Tied input/result encoder và query additive+residual đạt **5/30 (16.67%)**, so R08 **3/30** và hard negatives R02 **4/30**. Swap gap0, peak37.6MiB, code25.2s, notebook55.3s. Đây là ứng viên tốt nhất hiện tại trong điều kiện scratch; khác biệt một/hai câu trên30 mẫu chưa vững.

Một tên từng là input trong rule train có thể chuyển representation sang result nhờ cùng role space, thay vì học một projection result riêng. Query dùng a+b và residual nhỏ; đã đổi cả role tying và composition, không gán toàn bộ tác dụng cho một thành phần.

Independent cũng5/30, greedy4/30 trên score matrix này: lợi ích Hungarian phụ thuộc scorer. Không hứa Hungarian luôn tăng. 120 train rules không thay thế được world knowledge của BERT pretrained; không so ngang với reference pretrained khoảng0.41.

Đã nghiên cứu thêm hai ứng viên sau bảy lượt. Chưa đủ bằng chứng gọi đây là SOTA hoặc tăng ổn định; cần fold/seed khác trước kết luận mạnh.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
