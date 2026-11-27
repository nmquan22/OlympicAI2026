# Inverse Translation R09

R09 exact24/24, official100 và Kaggle100/100. Candidate evaluations2600 so7832 của R06 (giảm66.8%); measured code8.2827s so11.9711s (giảm30.8% ở hai lần đo). Search counter85 calls gồm16 gradient calls. Gradient sinh shortlist trong ngữ cảnh hiện tại rồi exact-verify margin; code vẫn có top8 fallback. Không có counter số lần fallback nên không quy mọi gain cho gradient riêng. Thời gian chịu warm-up/khác batch, chưa phải benchmark nhiều seed.

Canary bốn provided examples khớp ở mỗi run (8/8 checks trong batch), BATCH_RESULT có đủR08/R09. Peak 340.11MiB trên T4. Không sửa frozen model weights.

[5 paper và thiết kế](../../../research_v2/inverse-translation.md).

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
