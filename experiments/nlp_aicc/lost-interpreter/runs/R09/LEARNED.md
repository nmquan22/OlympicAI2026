# lost-interpreter R09

Typed input giảm lỗi ≤4 lệnh (5.91→5.17), nhưng tăng lỗi5–8/9–16 (17.53/23.00) và MAE tổng15.81 > R08 15.31. Public/private cũng kém R08. Chưa thay thế BPE BiGRU. Mượn typed inputs từ NPI, không có trace để reproduce NPI.

[Paper và thiết kế](../../../research_v2/lost-interpreter.md). Code 102.090s, peak 47.62MiB trên T4.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
