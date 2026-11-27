# word-lookups R09

BPE-dropout NLL5.78945 thấp hơn R08 nhưng Boundary F1.9145/.9168 thấp hơn.9150/.9171. NLL không thay được metric cuối. Giữ R08 làm preset tốt hơn trong lượt này; không tối ưu dropout tiếp theo private score.

[Paper và thiết kế](../../../research_v2/word-lookups.md). Code 244.072s, peak 981.78MiB trên T4.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
