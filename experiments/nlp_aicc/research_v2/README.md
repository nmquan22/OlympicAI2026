# Research vòng 2: từ lỗi thực tế đến thí nghiệm

Ngày bắt đầu: 27/11/2026. 41 cấu hình cũ đã gửi Kaggle: Moment 10, Lost 8, Word 7, Inverse 7, Alchemy 9. Điểm phải đọc theo source version + output path; mô tả submission có thể nhập sai. Hai mô tả Moment R02/R03 đầu đợt thực chất là source R05/R06; SCORES.csv ghi theo source thật.

Mỗi bài có 5 paper được chọn theo mức phù hợp với lỗi, khả năng train từ đầu, dữ liệu/supervision có sẵn và ngân sách T4. Đây là danh sách ưu tiên cho bài này, không phải bảng xếp hạng SOTA toàn ngành. Không dùng điểm private để chọn checkpoint. Các cấu hình vòng 2 được đóng băng trước khi nộp; đã nhìn thấy điểm cũ nên việc nghiên cứu tiếp vẫn có nguy cơ thích nghi leaderboard.

| Bài | Bằng chứng điểm nghẽn | Hướng kiểm tra |
|---|---|---|
| Moment Match | R04 val .6912, public .6436, private .6284; R07 public .6619 nhưng private .6162 | Học so sánh sau alignment; giữ tín hiệu cục bộ thay vì chỉ tăng self-attention |
| Lost Interpreter | R08 MAE val 15.3089, public 14.1608, private 16.7920; lỗi tăng theo số lệnh | Opcode/operand tách riêng; một recurrent step mỗi instruction; curriculum trộn độ khó |
| Word Lookups | R07 Boundary F1 .9097/.9113; không có nhãn segmentation train | Thống kê nhánh trái/phải và lexicon, không chọn threshold bằng test gold |
| Inverse Translation | Nhiều cấu hình đạt 100; R02/R03 bỏ lỡ một số target | Gradient proposal có exact verification; tối ưu chi phí ở cùng mức điểm |
| Alchemy | R09 val 5/30, public .0285/private .1142; phần lớn cách đổi loss vẫn gần ngẫu nhiên | Kiểm tra lexical coverage, square matching đúng giả định, shared attribute compatibility |

Mỗi lượt ghi paper nguồn, phần thực sự cài, phần không tái lập, điều kiện thất bại, metric và kết luận. Không gọi organizer reference là SOTA đã xác minh. Không ép một paper vào dữ liệu thiếu điều kiện của nó.

Thời gian: Lost mới tối đa 300 giây/config, Alchemy matching tối đa 180 giây/config; cộng khoảng 60 giây khởi động mỗi Saved Version. Dùng tối đa hai version song song; chỉ kiểm tra sau ETA hoặc sau khi hoàn thành một khối research. Thời gian queue tách khỏi thời gian train, không đoán là đã chạy.
