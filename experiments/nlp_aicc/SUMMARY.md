# Kết quả 5 bài NLP

Đã hoàn tất **41 thí nghiệm GPU T4 duy nhất** trên 5 bài, vượt mức tối thiểu35. Chúng chạy trong27 Saved Version, tối đa hai version đồng thời. Không tính config chạy lặp, lỗi cài package, canary lỗi hay retry thành một thí nghiệm mới. Audit: [AUDIT.json](AUDIT.json).

| Bài | Số thí nghiệm | Cấu hình chọn | Metric thật | Idea đáng giữ |
|---|---:|---|---|---|
| Moment Match | 10 | R04 | Validation accuracy **69.12%** /1,059 mẫu | IDF cho late interaction; thêm ba ứng viên research chưa vượt bản này |
| Lost Interpreter | 8 | R08 | Holdout MAE **15.31**, từ36.26 /3,000 mẫu | BPE BiGRU + packing/masking + SmoothL1; giảm57.8%, đổi cả backbone/loss |
| Word Lookups | 7 | R07 | Boundary F1 **public0.9097 / private0.9113** | Hai chiều + cohesion PMI + DP + dấu câu; từpublic0.7559 |
| Inverse Translation | 7 | R06 | Official **100, exact24/24**; code11.97s | Coordinate search giữ điểm, giảm9624→7832 candidate evaluations |
| Alchemy | 9 | R09 | Result-disjoint holdout **5/30 (16.67%)** | Shared input/result role space + commutative composition; mẫu nhỏ |

Các metric khác nhau, không lấy cột số để xếp hạng giữa các bài. Lost/Alchemy là validation, Word là late-submission test score sau khi đóng băng bảy preset, Inverse là evaluator gốc trên target đã được đề cung cấp. Một seed và30 mẫu Alchemy chưa đủ kết luận chắc. Reference pretrained và random-init là điều kiện riêng; không tuyên bố đạt SOTA.

## Ý tưởng đã được kiểm tra

- **Đừng gộp feature quá sớm:** Moment giữ tương tác token/event. IDF tăng65.44→69.12%, còn chỉ target center giảm49.10%. Context quan trọng, nhưng thêm Transformer hay word-feature fusion không tự tăng điểm.
- **Kiến trúc mới hơn không luôn hợp hơn:** Lost BiGRU có bias tuần tự phù hợp chương trình; model vẫn lỗi nhiều hơn khi chương trình dài. Không đồng nhất “đọc opcode” với “thực thi chính xác”.
- **Decoder có thể là nơi đáng cải thiện:** Word PMI tăngpublic0.7559→0.8766 dù NLL LM giữ nguyên. Bản kết hợp đạt0.9097. NLL mỗi BPE token không thay Boundary F1 và không so ngang tokenizer khác nhau.
- **Cùng điểm, tối ưu chi phí:** Inverse top8/top16 cắt mất proposal cần thiết; CE tốn41,464 candidate evaluations. Coordinate giữ100 với7,832. Restart chưa kích hoạt trên bộ target đã exact nên chưa có bằng chứng hữu ích.
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
