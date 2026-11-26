# NLP AICC · Học idea trước, đo kết quả sau

Theo yêu cầu người dùng, ưu tiên 5 bài dưới đây trước hàng đợi AICC tổng quát. **35 lượt T4** là mục tiêu, không phải số đã chạy. Bỏ qua các baseline thấp đã hiểu; chúng chỉ được phân tích từ code và số nguồn.

| Bài | Idea chính | Weights |
|---|---|---|
| [Moment Match](moment-match/IDEAS.md) | Late interaction và căn chỉnh chuỗi sự kiện | Chỉ dùng fastText được cấp trong dữ liệu. BPE subword vectors được khởi tạo từ các word vectors này; encoder/attention/CNN học mới. |
| [Lost Interpreter](lost-interpreter/IDEAS.md) | Biểu diễn chương trình, số và luồng điều khiển | Không dùng bất kỳ pretrained weights nào. |
| [Word Lookups](word-lookups/IDEAS.md) | Surprisal làm tín hiệu ranh giới và giải mã BMES | Không pretrained, external dictionary hoặc nhãn thủ công. Giữ test chỉ để inference; tách train holdout làm proxy LM, không gọi proxy đó là segmentation F1. |
| [Inverse Translation](inverse-translation/IDEAS.md) | Tìm kiếm rời rạc ngược qua một model có sẵn | Phải giữ model dịch, vocab/token IDs được cấp. Đổi tokenizer BPE mới sẽ phá ID của model, nên đây là ngoại lệ bắt buộc. |
| [Alchemy](alchemy/IDEAS.md) | Cross-encoder, negative sampling và ghép cặp toàn cục | Đề chỉ cho phép bert-base-uncased. Nhánh đúng checkpoint dùng tokenizer đi kèm (WordPiece), không tự thay thành BPE; mọi nhánh scratch cần ghi riêng điều kiện và không so ngang một cách gây hiểu nhầm. |

Mặc định BPE fit trên training split và model khởi tạo ngẫu nhiên. Riêng model/vocab do đề quy định giữ nguyên mapping: Inverse Translation dùng model dịch được cấp; Moment Match chỉ lấy fastText trong data; Alchemy chỉ dùng checkpoint BERT được đề cho phép, giữ tokenizer đi kèm. Không nạp checkpoint mới chỉ vì solution dùng nó.

Lời giải được dẫn nguồn là reference BTC. Chỉ gọi SOTA nếu có bằng chứng leaderboard/paper phù hợp cùng điều kiện. Điểm từ nguồn, validation tự chạy và leaderboard tự nộp luôn tách riêng. Các thay đổi là giả thuyết, không hứa chắc tăng điểm.

Sau **mỗi lượt hoàn tất** sẽ commit nhật ký vào repo, tối đa hai lượt Kaggle đang chạy. Mỗi báo cáo cần: input/output mẫu; flow baseline → reference → thay đổi; loss; cách decode; metric thật; chi phí; 3 nhóm lỗi; một câu kết luận học được gì.

