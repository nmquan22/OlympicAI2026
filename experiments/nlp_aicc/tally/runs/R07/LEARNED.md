# R07 — che toàn bộ từ hiếm để buộc dùng ngữ cảnh

Giữ count composite likelihood CRF của R03, không ghép nhánh character hoặc head biên của R06. Với từ xuất hiện đúng một lần trong **gold train + weak train**, thay cả pooled BPE vector bằng MASK embedding học từ đầu với xác suất 0,5. Áp dụng lúc warm-up/replay gold và weak training; giữ feature tần suất/độ dài, inference không che từ. Đây là regularization theo ý tưởng singleton replacement của Lample §4.1, không phải MLM pretraining.

| Phép đo | R03 | R07 |
|---|---:|---:|
| Validation span F1 | 0,444444 | 0,482759 |
| TP / FP / FN | 6 / 6 / 9 | 7 / 7 / 8 |
| PERSON đúng / gold | 4 / 7 | 5 / 7 |
| ORGANIZATION đúng / gold | 0 / 5 | 1 / 5 |
| LOCATION đúng / gold | 2 / 2 | 1 / 2 |
| Recall thực thể có từ chưa thấy | 6 / 14 | 7 / 14 |
| Weak count MAE / vector exact | 0,21975 / 36,125% | 0,22000 / 36,000% |
| Kaggle public / private | 0,35301 / 0,35300 | 0,38386 / 0,37104 |

PERSON tốt hơn nhưng có đánh đổi: `gymnassee city` bị cắt còn một từ; `amerida express` bị nhầm sang LOCATION. Trong câu nói về phim, model gán CREATIVE_WORK cho từ `project` và vẫn bỏ tên tác phẩm thật. CREATIVE_WORK đúng 0/1 dù đoán hai span; EVENT không có trong validation. Ba gold span sai biên cùng loại, năm span không overlap đúng loại. Che từ hiếm không giải quyết được toàn bộ định vị và phân loại.

Có **399.868 / 799.751 lượt từ đủ điều kiện bị che** trong quá trình train; đây là số observations có lặp qua batch/epoch, không phải số từ unique. Không dùng nhãn hoặc count test.

Code chạy **480,813 giây**, peak **135,396 MiB**, tổng Kaggle **506,7 giây**. Best checkpoint ở epoch 20, tổng 3.324 optimizer steps. Cùng cap với R03 nhưng số bước/early stopping khác; một seed chưa chứng minh riêng tác dụng masking ổn định.

**Bài học:** zero UNK của byte BPE không đảm bảo tổng quát hóa tên lạ. Che cả từ là phép thử rẻ để khuyến khích dùng context. R07 dẫn public trong bảy cấu hình, nhưng validation thấp hơn R06; báo riêng hai tiêu chí, không chọn theo private và không tuyên bố SOTA.

Cả CRF/count/teacher và boundary/singleton canary đã pass trên T4 trước train. Đã nộp đúng `tally/R07/submission.csv`, Saved Version 7/script374948086, Complete(after deadline). Source bất biến `469e5332a09dd0faf6bcf7970cdf7c1b33d33d02`.

[Nguồn năm paper và cách chuyển ý tưởng](../../RESEARCH.md) · [So sánh cả bảy lượt](../../RESULTS.md).

Bằng chứng được đọc từ rendered output Kaggle thật. Repo giữ summary, best/final epoch và hai trong bảy error cases; không phải full stdout hoặc notebook đã execute. Toàn bộ history/cases ở version trong evidence.json.
