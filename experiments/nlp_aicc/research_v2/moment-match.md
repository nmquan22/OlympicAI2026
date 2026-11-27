# Moment Match: so sánh có cấu trúc sau alignment

R04 IDF tốt nhất val (.6912) nhưng Kaggle .6436/.6284. R07 token contextualization + IDF public .6619/private .6162. R08 monotonic prior không vượt R04 val; R10 ghép thêm intact supplied-word channels còn giảm. Thêm feature không bảo đảm giữ được optimization và inductive bias tốt. R10 đổi số kênh CNN nên cũng đổi initialization; không quy hết giảm điểm cho word features.

| Hạng | Paper | Chuyển sang bài này | Không chuyển máy móc |
|---|---|---|---|
| 1 | [A Decomposable Attention Model for NLI, Parikh et al., EMNLP 2016](https://aclanthology.org/D16-1244.pdf) §2 | Attend → compare → aggregate; học phần khác nhau sau khi đã ghép tương ứng | NLI sentence-pair khác chọn timeline; dùng event vectors, không nhận mình tái lập SNLI |
| 2 | [ColBERT, Khattab & Zaharia, SIGIR 2020](https://arxiv.org/html/2004.12832v2) §3 | Giữ token-level late interaction để tránh ép cả câu thành một vector | Không tải BERT; chỉ supplied fastText đã có trong đề. Hard MaxSim R09 đã thử, không mặc định tốt hơn soft |
| 3 | [Text Matching as Image Recognition / MatchPyramid, Pang et al., AAAI 2016](https://arxiv.org/pdf/1602.06359) §MatchPyramid | CNN trên interaction grid học mẫu khớp cục bộ | Ở đây grid event5×9, không cùng grid word bài gốc; giữ nhánh cũ khi thêm reasoning |
| 4 | [Drop-DTW, Dvornik et al., NeurIPS 2021](https://arxiv.org/html/2108.11996v1) §3 | Timeline chứa event không tương ứng; cần cho phép bỏ nhiễu | Chưa cài Drop-DTW đầy đủ. R08 strict subsequence không phải Drop-DTW; nếu full matching gây hại thì đây là nhánh kế |
| 5 | [Soft-DTW, Cuturi & Blondel, ICML 2017](https://proceedings.mlr.press/v70/cuturi17a.html) | Soft-min recurrence cho phép path lặp/căn chỉnh co giãn, gradient truyền qua alignment | Full DTW vẫn ép mọi event phải match; có thể hại nếu nhiều outlier |

**R11** giữ R04 grid branch, thêm nhánh event-level decomposable attention: IDF-weighted pooling BPE → MLP attention → soft alignment 5×9 theo hai chiều → compare [x,aligned,|difference|,product] → aggregate → residual score với scale học được. Đây là cross-attention phụ thuộc candidate, khác self-attention trước khi so sánh ở R03. Không xử lý token như ảnh, không lấy CV weights.

**R12** giữ R04, thêm residual negative Soft-DTW cost trên grid similarity. Recurrence có diagonal/up/left, soft-min gamma .1; khác R08 strict-increasing subsequence. Canary finite gradient và toy all-matched. Mục đích phân biệt bias alignment tốt với ép match nhiễu; không kỳ vọng paper nổi tiếng tự làm điểm tăng.

Cùng seed42/train/provided validation, tối đa30 epoch/240 giây mỗi config. Chọn checkpoint bằng validation. Lưu lỗi tự tin cao để đọc câu target, candidate sai và candidate đúng. Chưa có kết quả GPU vòng 2 khi chốt thiết kế.
