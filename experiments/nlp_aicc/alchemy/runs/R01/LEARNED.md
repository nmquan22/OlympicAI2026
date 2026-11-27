# R01 · Matching toàn cục xử lý va chạm candidate

Scratch BPE cross-encoder đạt **3/30 (10%)** với Hungarian, **1/30** với argmax độc lập, greedy cũng 3/30. Argmax tạo **19 lựa chọn trùng**. Trên cùng score matrix, ràng buộc một-một có ích nhưng chưa bảo đảm hơn greedy.

Mỗi input là [CLS] item1 [SEP] item2 [SEP] result; encoder thấy candidate khi tạo feature. Train CE sampled candidate + swap consistency. 120 rule train, 30 result-disjoint holdout; toàn bộ model khởi tạo mới, không dùng BERT checkpoint.

Swap score gap vẫn 0.70: penalty xác suất nhỏ chưa đảm bảo đối xứng logits. Mẫu quá ít để học kiến thức “thế giới”; không so ngang score với reference BERT pretrained. Sẽ thử hard negatives, full-list CE, depth, probability averaging và Sinkhorn.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
