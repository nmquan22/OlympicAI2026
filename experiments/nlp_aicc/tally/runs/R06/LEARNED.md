# R06 — học thêm vị trí bắt đầu và kết thúc thực thể

Giữ encoder BPE → mean pool từng từ → BiGRU và count composite likelihood của R03. Thêm head dự đoán start/end cho năm loại từ hidden của BiGRU. Gold loss = CRF NLL chia độ dài câu + 0,1 × balanced BCE cho biên; bỏ padding. Hai head phụ chỉ dùng khi train, inference vẫn là Viterbi của IO CRF sáu nhãn. Đây **không phải semi-CRF**.

| Phép đo | R03 | R06 |
|---|---:|---:|
| Validation span F1 | 0,444444 | 0,571429 |
| TP / FP / FN | 6 / 6 / 9 | 8 / 5 / 7 |
| ORGANIZATION đúng / gold | 0 / 5 | 2 / 5 |
| Recall thực thể có từ chưa thấy | 6 / 14 | 8 / 14 |
| Weak holdout count MAE | 0,21975 | 0,21975 |
| Đúng toàn vector count | 36,125% | 35,125% |
| Kaggle public / private | 0,35301 / 0,35300 | 0,34627 / 0,35248 |

R06 nhận đúng cả `new dowmensus university`, nhưng vẫn bỏ `graduate school in public service`. Nó còn cắt thiếu `stock exchange`, `amerida express` và tên người chứa gạch nối. CREATIVE_WORK vẫn đúng 0/1; validation không có EVENT nên không đánh giá được loại này. Bốn gold span còn overlap đúng loại nhưng sai biên, ba span không có overlap đúng loại.

Code chạy **490,922 giây**, peak **135,422 MiB**, Kaggle hiển thị tổng **520,4 giây**. Best checkpoint ở epoch 26; tổng 3.381 optimizer steps so với 2.533 của R03. Cùng trần train 480 giây nhưng early stopping và trajectory khác; khởi tạo thêm head cũng làm RNG tiêu thụ khác. Một seed chưa cô lập được hiệu ứng auxiliary loss khỏi thời gian và trajectory huấn luyện.

**Bài học:** supervision biên có tín hiệu trên pilot, nhưng validation cao nhất không chuyển thành public cao nhất. Tập 10 câu quá nhỏ để chọn model chắc chắn. BEST_CONFIGS ghi R06 theo validation, còn R07 dẫn public được báo riêng. Private chỉ báo cáo, không dùng chọn nâng cấp.

Cả CRF/count/teacher canary và boundary/singleton representation canary đã pass trên T4 trước train. Đã nộp đúng `tally/R06/submission.csv`, Saved Version 6/script374947843, Complete(after deadline). Source bất biến `469e5332a09dd0faf6bcf7970cdf7c1b33d33d02`.

[Nguồn năm paper và cách chuyển ý tưởng](../../RESEARCH.md) · [So sánh cả bảy lượt](../../RESULTS.md).

Bằng chứng được đọc từ rendered output Kaggle thật. Repo giữ summary, best/final epoch và hai trong bảy error cases; không phải full stdout hoặc notebook đã execute. Toàn bộ history/cases ở version trong evidence.json.
