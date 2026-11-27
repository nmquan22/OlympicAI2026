# Tally R01 · expectation chưa giải được vị trí và biên

T4 code148.875s, peak92.439MiB. ValF1=0.444444 (TP6,FP6,FN9;15entities/10câu). Kaggle public0.29710/private0.30080, Complete(afterdeadline), đúng tally/R01/submission.csv, version1/script374844445. Private chỉ ghi nhận.

ORGANIZATION đúng0/5, chỉ dự đoán1span; bỏ nhiều tổ chức và cắt `new matrellah stock exchange` thành `new matrellah`. Ba gold span cùng loại bị sai biên, năm không có overlap đúng loại, một đúng biên sai loại. `dickett kennelson - mccarl` bị cắt trước gạch nối: punctuation bên trong span là hợp lệ. Recall entity chứa từ chưa gặp6/14;0UNK chưa có nghĩa model hiểu từ. Weak countMAE0.22375, đúng cả vector31.5%; không gọi count đã tốt.

Train loss giảm0.03060→0.01773 từepoch12→20 nhưng valF1 giảm0.4444→0.3077, nên giữ early checkpoint. Validation không cóEVENT và chỉ1CREATIVE_WORK, không có bằng chứng tổng quát mọi loại.

Lample NER và GE gợi ý CRF, dùng pair marginals để học kỳ vọng số lần bắt đầu tên; Measurements gợi ý count likelihood qua DP thay cho chỉ khớp trung bình. R02 sẽ thử cấu trúc CRF; R03 tách riêng tác động likelihood count. Tổng năm log marginal chỉ là composite likelihood, không joint count likelihood. Giữ split/encoder/maximum budget, log runtime thật và không hứa tăng. [Năm paper](../../RESEARCH.md).


Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
