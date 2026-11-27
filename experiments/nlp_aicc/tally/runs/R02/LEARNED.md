# Tally R02

CRF pair marginals/Viterbi chưa sửa bottleneck: valF1 vẫn6TP/6FP/9FN và ORG0/5, countMAE còn .23000 so R01 .22375. Public .30394 tăng nhỏ so .29710, nhưng code335.166s so148.875s (~2.25 lần). Không gọi đây là cải thiện hiệu quả. Tên stock exchange còn bị rút từ2 từ xuống1 từ; transition không bổ sung đủ ngữ cảnh/ngữ nghĩa.

T4 code335.165988s, peak92.444824MiB; Kaggle tổng363.4s. Canary partition/count expectation/count numerator/Viterbi/padding/finite gradients đều pass trước train; BATCH_RESULT đầy đủ.

Giả thuyết tiếp: character CNN hỗ trợ tên hiếm (14/15 gold entity chứa từ chưa gặp), và PR teacher điều chỉnh posterior theo năm count đồng thời. PR phải log residual thực, không gọi moment matching hoặc teacher chưa hội tụ là projection chính xác. [Năm paper](../../RESEARCH.md).

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.
