# Kết quả 5 bài NLP

Đã hoàn tất và nộp Kaggle **51 cấu hình T4 duy nhất** trên 5 bài, vượt mức tối thiểu35. Mười cấu hình mới trong năm batch được thiết kế sau khi lọc [5 paper cho mỗi bài](research_v2/README.md). [Kết quả và bài học vòng2](research_v2/RESULTS.md). Không tính config chạy lặp, lỗi cài package, canary lỗi hay retry thành một thí nghiệm mới. Audit: [AUDIT.json](AUDIT.json).

| Bài | Số thí nghiệm | Cấu hình chọn | Metric thật | Idea đáng giữ |
|---|---:|---|---|---|
| Moment Match | 12 | R12 theo val | Validation **69.41%**; Kaggle **.6456/.6253** | Soft-DTW tăng3/1059 câu val nhưng chưa tăng private; chưa khẳng định ổn định |
| Lost Interpreter | 10 | R08 | Holdout MAE **15.31**; Kaggle **14.1608/16.7920** | Typed instructions và curriculum mới chưa vượt BiGRU BPE |
| Word Lookups | 9 | R08 | Boundary F1 **public.9150 / private.9171** | Accessor-variety spans tăng từR07 .9097/.9113 ở cùng LM recipe |
| Inverse Translation | 9 | R09 | Official **100, exact24/24**; Kaggle **100/100**; code8.28s | Contextual gradient proposals + exact verification; 2600 candidates so7832 của R06 |
| Alchemy | 11 | R09 | Holdout **5/30**; Kaggle **.0285/.1142** | Square matching không tăng; 24/30 tên val chưa thấy trong train dù0 UNK |

Các metric khác nhau, không lấy cột số để xếp hạng giữa các bài. Điểm Kaggle ghi public/private; mỗi preset đóng băng trước lần chấm của nó. Inverse đã xác nhận100/100 trên Kaggle. Một seed và30 mẫu Alchemy chưa đủ kết luận chắc. Reference pretrained và random-init là điều kiện riêng; không tuyên bố đạt SOTA. Không dùng private để chọn checkpoint.

## Ý tưởng đã được kiểm tra

- **Đừng gộp feature quá sớm:** Moment giữ tương tác token/event. IDF tăng65.44→69.12%, còn chỉ target center giảm49.10%. Context quan trọng, nhưng thêm Transformer hay word-feature fusion không tự tăng điểm.
- **Kiến trúc mới hơn không luôn hợp hơn:** Lost BiGRU có bias tuần tự phù hợp chương trình; model vẫn lỗi nhiều hơn khi chương trình dài. Không đồng nhất “đọc opcode” với “thực thi chính xác”.
- **Decoder có thể là nơi đáng cải thiện:** Word PMI tăngpublic0.7559→0.8766 dù NLL LM giữ nguyên. Bản kết hợp đạt0.9097. NLL mỗi BPE token không thay Boundary F1 và không so ngang tokenizer khác nhau.
- **Cùng điểm, tối ưu chi phí:** Inverse R09 giữ100 với2,600 candidate evaluations so7,832 của coordinate R06. R08 gradient fallback sau top8 chỉ đạt23/24, không chọn dù đã giảm candidates. R09 dùng gradient theo ngữ cảnh hiện tại và exact-verify; thời gian8.28s so11.97s là đo một lần, chịu warm-up/khác batch. Restart cũ chưa kích hoạt trên bộ target đã exact nên chưa có bằng chứng hữu ích.
- **Ràng buộc và representation cùng quan trọng:** Alchemy Hungarian xử lý candidate trùng, symmetry có thể đảm bảo bằng kiến trúc; role tying tăng3/30→5/30 trong lần chạy này. 120 rule train chưa thay thế được kiến thức BERT pretrained.

[LEARNING_GUIDE.md](LEARNING_GUIDE.md) giải thích input→baseline→reference→nâng cấp. [RESEARCH_NEXT.md](RESEARCH_NEXT.md) ghi nguồn paper, giả thuyết và giới hạn. Mỗi `runs/Rxx/LEARNED.md` có số và kết luận riêng. [BEST_CONFIGS.json](BEST_CONFIGS.json) chỉ rõ notebook version, source SHA và đường dẫn submission của các ứng viên đã chọn.

## Mốc kiểm tra GPU

Ước lượng từ **thời gian code đã đo +60s khởi động**, không gồm queue:

| Job | Lần đầu nên kiểm tra |
|---|---:|
| Một lượt Moment Match | Khoảng3 phút |
| Batch Lost R06–R08 | Khoảng7 phút |
| Batch sáu lượt Word R02–R07 | Khoảng16 phút |
| Batch bảy lượt Inverse | Khoảng3 phút |
| Một lượt Alchemy | Khoảng2 phút |

`python tools/nlp_eta.py --task word-lookups --runs 2 3 4 5 6 7` tính lại mốc từ log. Khi còn queue, giãn lần kiểm tra2→4→8→15 phút. Trong lúc chờ, đọc solution, phân tích lỗi và viết báo cáo; không mở trạng thái từng phút. Batch giữ model/optimizer/seed và output folder riêng cho từng config.

## Giới hạn thực tế

Các checkpoint Lost/Alchemy được chọn từ training fold, chưa refit toàn train. Bảy preset Word đã cố định trước khi xem public/private score; không fit threshold bằng gold test. Phần lớn báo cáo lưu trường đo đọc từ output Kaggle cùng URL/source SHA, một số có notebook execute đầy đủ; `evidence.json` phân biệt rõ, không giả lập stdout hay notebook. Audit kiểm tra chứng cứ/schema/count, không chạy lại toàn bộ training và không chứng minh ý nghĩa thống kê.

Kiểm tra code cuối còn phát hiện regex numeric ở các source cũ có thể lẫn chữ số OP11/OP12. Đã sửa và kiểm tra operand parsing; R03/R07 ghi rõ hạn chế. Không đổi số lịch sử, chưa gán score mới cho bản sửa; R08 tốt nhất không dùng numeric branch.


## Bài thứ sáu đang chạy: Tally

Tổng51 kết quả và51 submissions ở trên thuộc năm bài đã hoàn tất. [Tally](tally/IDEAS.md) mới có R01 Saved Version1 đang queued T4x2, chưa có metric hay submission. Mục tiêu hiện tại là thêm ít nhất7 cấu hình có ý tưởng khác nhau; xem [STATUS.md](STATUS.md) và [trạng thái Tally](tally/STATE.json). Audit đã kiểm bằng chứng51 lượt cũ không có lỗi schema, nhưng mục tiêu toàn workflow còn pending vì Tally chưa đủ bảy. Không cộng job đang queue vào tổng thành công.


### Tally R01 có kết quả

R01 đã hoàn tất T4: code148.875s, peak92.439MiB; val spanF1.444444 trên10câu (TP6/FP6/FN9), Kaggle public.29710/private.30080, Complete(afterdeadline). Tổng hiện52 cấu hình hoàn tất/scored; Tally mới1/7. Lỗi chính: bỏ tổ chức, hụt biên tên; weak countMAE.22375 và31.5% đúng vector. [Nhật ký R01](tally/runs/R01/LEARNED.md); R02/R03 sẽ thử CRF pair expectations và count composite likelihood sau canary. Các đoạn51 phía trên là snapshot năm bài cũ.


### Tally R02/R03 đã hoàn tất và nộp

Tổng54 cấu hình hoàn tất/scored, Tally3/7. R02 CRF expectations: val.444444, public.30394/private.30351,335.166s/92.445MiB. R03 count composite likelihood: val.444444, public.35301/private.35300,374.102s/136.328MiB. Cả hai pass exhaustive CRF canary. R03 weak exact count vector36.125% so31.5% củaR01, nhưng ORG vẫn0/5 và lỗi biên cùng loại5 so3. Xem [R02](tally/runs/R02/LEARNED.md), [R03](tally/runs/R03/LEARNED.md).

### Tally R04/R05: hai kết luận khác nhau

Tally5/7, tổng56 cấu hình hoàn tất/scored. R04 byte character CNN: val.428571, public.37057/private.37740,330.592s/156.699MiB. Public cao hơnR03 .35301 nhưng val thấp hơn; mười câu chưa đủ chọn ổn định. R05 approximate teacher KL: val.370370, public.25020/private.25609,490.202s/92.551MiB. Train-batch countMAE .270172→.159125 là chẩn đoán projection, không phải valF1; holdout countvector29.25% kémR03 36.125%. Giữ kết quả âm. [R04](tally/runs/R04/LEARNED.md), [R05](tally/runs/R05/LEARNED.md). Tiếp theo R06 auxiliary biên và R07 singleton masking, chưa có kết quả.
