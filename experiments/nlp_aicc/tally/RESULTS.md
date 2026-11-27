# Tally · kết quả bảy lượt luyện tập

Hoàn tất ngày28/11/2026: **7 cấu hình T4 / 7 submissions Complete(after deadline)**. Tổng sáu bài hiện58 cấu hình/scored; năm bài cũ51 giữ nguyên. Không có job đang chạy, không tính retry là cấu hình mới.

| Run | Ý tưởng | Val spanF1 | Public | Private | Code(s) | Peak(MiB) |
|---|---|---:|---:|---:|---:|---:|
| [R01](runs/R01/LEARNED.md) | Independent expected run counts | 0.444444 | 0.29710 | 0.30080 | 148.875 | 92.439 |
| [R02](runs/R02/LEARNED.md) | CRF pair-marginal expected counts | 0.444444 | 0.30394 | 0.30351 | 335.166 | 92.445 |
| [R03](runs/R03/LEARNED.md) | Per-type count composite likelihood | 0.444444 | 0.35301 | 0.35300 | 374.102 | 136.328 |
| [R04](runs/R04/LEARNED.md) | R03 + byte character CNN | 0.428571 | 0.37057 | 0.37740 | 330.592 | 156.699 |
| [R05](runs/R05/LEARNED.md) | Approximate moment teacher KL | 0.370370 | 0.25020 | 0.25609 | 490.202 | 92.551 |
| [R06](runs/R06/LEARNED.md) | R03 + typed boundary auxiliary loss | 0.571429 | 0.34627 | 0.35248 | 490.922 | 135.422 |
| [R07](runs/R07/LEARNED.md) | R03 + singleton whole-word masking | 0.482759 | 0.38386 | 0.37104 | 480.813 | 135.396 |

Tổng code của7 lượt: 2650.672s. Tổng runtime hiển thị Kaggle: 2852.4s; không gồm queue. Được cấp GPU T4x2 mỗi job; đây không phải số GPU-giờ tính quota đã kiểm chứng.

## Chọn model và bài học

**R06 dẫn validation 0.571429**, public 0.34627. **R07 dẫn public 0.38386**, validation 0.482759. Private chỉ lưu để báo cáo; không dùng chọn architecture hoặc checkpoint. Public R07 tăng từ R01 0.29710 lên 0.38386, tương đương **8.676 điểm phần trăm**. Tập validation chỉ 10 câu nên chưa có lựa chọn tốt nhất ổn định; không tuyên bố SOTA.

| Câu hỏi nghiên cứu | Kết quả đã quan sát | Bài học |
|---|---|---|
| Matching expectation hay likelihood của count? | R02 public 0.30394 → R03 0.35301; đúng toàn vector count 31.125% → 36.125%. | Likelihood có ích trong loạt này, nhưng count không xác định vị trí/độ dài tên. Năm DP riêng chỉ cho composite likelihood. |
| Zero UNK đã đủ cho tên giả? | R07 PERSON đúng 5/7 so với 4/7 của R03; recall thực thể có từ chưa thấy 7/14 so với 6/14. LOCATION giảm 2/2 → 1/2. | Che cả pooled word vector của singleton là regularization khuyến khích dùng context, vẫn có đánh đổi loại thực thể. |
| Thêm supervision cho biên có giúp? | R06 ORGANIZATION đúng 2/5 so với 0/5; count MAE vẫn 0.21975. Val tốt nhất nhưng public thấp hơn R03. | Tín hiệu biên có ích trên pilot, chưa tổng quát chắc. Head phụ chỉ dùng train, không phải semi-CRF. |
| Teacher khớp moment hơn có chắc tăng F1? | R05 giảm mean train-batch count MAE 0.270172 → 0.159125 nhưng public chỉ 0.25020. | Residual và canary là chẩn đoán tính toán, không thay metric đích hoặc chứng minh objective học tốt. |

Chi phí và duration cũng là biến: R06/R07 dùng 3.381/3.324 optimizer steps so với 2.533 của R03. Cùng cap 480 giây nhưng early stopping, trajectory và khởi tạo phần thêm khác nhau. Một seed chưa cô lập được tác dụng auxiliary loss hoặc masking khỏi các yếu tố đó.

## Failure modes còn lại

R06 nhận đúng `new dowmensus university`, nhưng bỏ `graduate school in public service`. Nó còn cắt thiếu `stock exchange`, `amerida express` và tên người có gạch nối. Bốn gold span overlap đúng loại nhưng sai biên, ba span không overlap đúng loại. CREATIVE_WORK đúng 0/1.

R07 sửa một số tên PERSON nhưng nhầm `amerida express` thành LOCATION, cắt `gymnassee city` còn một từ. Trong câu về phim, nó nhận `project` là CREATIVE_WORK và bỏ tên tác phẩm thật. Ba gold span sai biên cùng loại, năm span không có overlap đúng loại. Cả hai không đánh giá được EVENT vì validation không có entity loại này.

Đã che 399.868 trong 799.751 **lượt từ đủ điều kiện** khi train R07. Các từ được quan sát lặp qua batch/epoch; không gọi đây là số từ unique. Inference không che từ, không có count test để điều kiện hóa decoder.

## Baseline, năm paper và giới hạn

BTC dùng pretrained ELECTRA, first-subword CE trên 50 gold trong 400 epochs, không dùng weak counts. Đây là điều kiện khác scratch; không so như cùng tài nguyên. Pipeline ở đây là train-only byte BPE → mean pool về từ → BiGRU → IO head/CRF, học 40 gold + 7.200 weak counts. Không weights/tokenizer/dữ liệu ngoài; 4.000 câu test chỉ để inference. Chưa refit toàn bộ 50 gold. Dấu câu được phép nằm trong span, không cấm toàn bộ punctuation.

[Năm paper và cách chuyển ý tưởng](RESEARCH.md): GE gợi ý expectation matching; Measurements gợi ý count likelihood; PR dẫn đến teacher KL; Lample dẫn đến CRF, character và singleton replacement; Sarawagi giúp xem đơn vị dự đoán là span. R06 chỉ chuyển ý tưởng biên thành loss phụ, chưa triển khai semi-CRF nguyên bản. IOAI correction chưa đọc vì membership; không nhận đã tái lập reference.

Nếu mở một đợt nghiên cứu mới: ưu tiên kiểm nhiều split/seed trên 50 gold với holdout riêng, rồi mới thử semi-Markov head chấm start/end/pool/length để xử lý hụt biên. Mỗi DP mới cần brute-force partition, Viterbi, count và finite-gradient canary. Đây là đề xuất **chưa chạy**; không tự thêm R08 trong batch đã hoàn tất.

## Bằng chứng

Mỗi run có report/evidence/LEARNED/failure_analysis. Repo giữ selected rendered summary, best/final epoch và hai trong bảy error cases của R06/R07; không phải full stdout hoặc executed notebook. Toàn bộ history/error cases còn ở đúng Kaggle version trong evidence. Launch notebooks trong repo vẫn có execution_count null và outputs rỗng.

CRF/count/teacher và representation canary đều đã quan sát pass trên T4 trước train R06/R07. Cả hai submission có 4.000 dòng, qua kiểm schema, unique IDs, word-tag alignment và allowed labels. Đã kiểm notebook version và output file trước Submit, rồi xác nhận Complete(after deadline) cùng điểm public/private.

[SCORES.csv](SCORES.csv) · [Audit](../AUDIT.json) · [Source code](../code/tally.py) · [R06](runs/R06/LEARNED.md) · [R07](runs/R07/LEARNED.md).
