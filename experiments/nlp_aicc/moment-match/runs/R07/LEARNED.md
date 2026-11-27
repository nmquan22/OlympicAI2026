# R07 · Hai cải tiến riêng lẻ không cộng điểm tự động

Token attention + IDF đạt **66.48%**, thấp hơn IDF R04 **69.12%**. Đủ bảy thí nghiệm, chọn R04 theo validation. Best epoch 5, về sau train loss giảm nhưng validation giảm.

Tám mẫu tự tin sai được đọc từ output cho thấy ba nhóm đáng điều tra: hành động lặp (paint/dip) làm nhiều đoạn có từ giống nhau; cùng nội dung nhưng khác offset/thứ tự sự kiện; paraphrase/đồ vật hiếm. Ví dụ row 208 có confidence 0.994 nhưng chọn đoạn lặp paint/dip khác đoạn gold. Không suy luận gold sai chỉ từ văn bản.

Idea mới R08: thêm prior alignment tăng đơn điệu trên 5×9 event grid, giữ IDF của R04. Phân biệt “trộn context” với “ràng buộc đường căn chỉnh”: R03 thất bại không bác bỏ prior thứ tự. R09 sẽ thử max alignment gần reference thay soft alignment. Chưa chạy hai ý tưởng này, chưa có số.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
