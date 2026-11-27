# Năm paper ưu tiên theo độ sát Tally

Đây là shortlist theo **mức liên quan**, không phải xếp hạng citation hay năm paper mới nhất. Đã đối chiếu nội dung nguồn chính: GE/Measurements/PR về giám sát gián tiếp, semi-CRF về suy luận span, Lample về biểu diễn và CRF. Các hướng dưới đây là adaptation cho Tally, không nhận là bản tái lập nguyên paper hoặc đã đạt số benchmark của paper. Kết luận thực nghiệm còn trống cho đến khi R01 hoàn tất.

| Ưu tiên | Nguồn chính | Điểm lấy được | Điều kiện để đưa vào experiment |
|---|---|---|---|
| 1 | Mann & McCallum, ACL2008, [Generalized Expectation Criteria for Semi-Supervised Learning of CRFs](https://aclanthology.org/P08-1099/) · §4 | Đưa thông tin về expectation vào objective của model chuỗi. Bản gốc dùng phân phối label theo feature; Tally cung cấp count mỗi câu. | R01 dùng squared discrepancy của run-start expectation. Đây là biến thể GE, không phải KL feature-label nguyên bản. Nếu count khớp nhưng span sai, cần objective/cấu trúc mạnh hơn. |
| 2 | Liang, Jordan & Klein, ICML2009, [Learning From Measurements in Exponential Families](https://icml.cc/2009/papers/393.pdf) · §2–4 | Nhãn đầy đủ và thống kê gián tiếp có thể xem là phép đo khác nhau của output ẩn. Bài có ví dụ đo transition vào block và dùng variational approximation. | Gợi ý likelihood của count thay vì chỉ matching mean. DP count riêng mỗi loại cho composite likelihood là adaptation; không gọi nó joint likelihood của vector năm count hoặc Bayesian inference đầy đủ. |
| 3 | Ganchev et al., JMLR2010, [Posterior Regularization for Structured Latent Variable Models](https://www.jmlr.org/papers/v11/ganchev10a.html) · §2.2–2.6 | Project posterior vào tập thỏa ràng buộc expectation bằng KL; khi feature cùng factorization có thể giữ suy luận tractable. | Nếu pseudo-label tự củng cố lỗi, thử teacher bị ràng buộc count trên weak train và student học KL. Cần giải dual/projection thật; thêm count MSE không được đổi tên thành PR. Không có count test để project khi inference. |
| 4 | Sarawagi & Cohen, NeurIPS2004, [Semi-Markov CRFs for Information Extraction](https://proceedings.neurips.cc/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html) · §2.1–2.2 | Chấm điểm segment thay vì từng token, dùng Viterbi theo vị trí, loại, độ dài tối đa. | Khi lỗi biên/span dài chiếm ưu thế: head span dùng start/end/pool/length. Phải báo coverage nếu giới hạn độ dài; không tuyên bố cùng model nếu chỉ hậu xử lý token scores. |
| 5 | Lample et al., NAACL2016, [Neural Architectures for Named Entity Recognition](https://aclanthology.org/N16-1030/) · §2, §4 | Kết hợp biểu diễn theo ký tự/ngữ cảnh; CRF chấm chuỗi nhãn chung; dropout giúp không chỉ dựa vào một nhánh biểu diễn. | Khi tên giả/OOV hoặc nhãn láng giềng yếu: thêm char encoder và/hoặc CRF có ablation riêng. Bản gốc dùng embedding học từ corpus khác; bản Tally sẽ bỏ tài nguyên đó và train từ đầu. Byte BPE hiện tại không đồng nghĩa đã có character BiLSTM của paper. |

## Hai phép phân biệt cần học

**Expectation matching và count likelihood khác nhau.** Model có thể chia xác suất giữa một chuỗi có 0 và một chuỗi có 2 tên để expectation bằng1; điều này không bảo đảm có khối lượng xác suất cao ở chuỗi có đúng1 tên. Đây là lý do đáng thử count-augmented DP nếu R01 biểu hiện xác suất loãng. Dùng một DP riêng từng loại tránh state tích Descartes lớn, nhưng tổng năm log marginal chỉ là composite likelihood.

**Chuỗi và span khác nhau.** CRF token có transition giúp nhãn nhất quán. Semi-CRF còn có thể chấm cả một span bằng feature độ dài/biên. Một span bị hụt một từ là sai hoàn toàn theo metric Tally, dù token accuracy còn cao. Đánh giá bằng span F1 quyết định có giữ nâng cấp hay không.

## Quy trình chọn lượt tiếp

1. Đọc `TALLY_FAILURE_ANALYSIS` và toàn bộ mười câu validation, xem TP/FP/FN và support từng loại; không chỉ nhìn một scalar F1.
2. Chọn một bottleneck đo được, ghi ví dụ lỗi, rồi chọn paper ở bảng. Đóng băng hypothesis và nguồn trước khi submit.
3. Giữ seed/split/nguồn dữ liệu, log runtime và VRAM. Nếu thay chi phí hay budget phải nêu rõ.
4. Nộp output chính xác từ Saved Version hoàn tất, đọc public/private score và giữ cả trường hợp thất bại. Không dùng private score để chọn hyperparameter.
5. Lượt sau lấy lỗi còn lại làm đầu vào. Chỉ sau khi có ít nhất bảy cấu hình hoàn tất mới ghi đạt mục tiêu bảy; nếu không cải thiện thì ghi thất bại và nghiên cứu tiếp trong phạm vi yêu cầu.
