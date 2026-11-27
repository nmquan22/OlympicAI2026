# NLP AICC · Học idea trước, đo kết quả sau

Đọc [LEARNING_GUIDE.md](LEARNING_GUIDE.md) để nắm idea; [STATUS.md](STATUS.md) để xem số đã đo. Mỗi `runs/Rxx/LEARNED.md` có kết luận và giới hạn của thí nghiệm.

Theo yêu cầu người dùng, ưu tiên 5 bài dưới đây trước hàng đợi AICC tổng quát. Năm bài đầu đã hoàn tất51 thí nghiệm T4 duy nhất, mỗi bài ít nhất7. Tally đã hoàn tất và nộp đủ 7 lượt, nâng tổng lên 58; R07 dẫn public F1 0.38386, R06 dẫn validation 0.571429. [Báo cáo Tally](tally/RESULTS.md). Chi tiết ở [SUMMARY.md](SUMMARY.md), [bảng metric](EXPERIMENTS.csv), [cấu hình đã chọn](BEST_CONFIGS.json) và [audit](AUDIT.json). Lượt lỗi/chạy lặp không tính thêm. Bỏ qua các baseline thấp đã hiểu; chúng chỉ được phân tích từ code và số nguồn.

| Bài | Idea chính | Weights |
|---|---|---|
| [Moment Match](moment-match/IDEAS.md) | Late interaction và căn chỉnh chuỗi sự kiện | Chỉ dùng fastText được cấp trong dữ liệu. BPE subword vectors được khởi tạo từ các word vectors này; encoder/attention/CNN học mới. |
| [Lost Interpreter](lost-interpreter/IDEAS.md) | Biểu diễn chương trình, số và luồng điều khiển | Không dùng bất kỳ pretrained weights nào. |
| [Word Lookups](word-lookups/IDEAS.md) | Surprisal làm tín hiệu ranh giới và giải mã BMES | Không pretrained, external dictionary hoặc nhãn thủ công. Giữ test chỉ để inference; tách train holdout làm proxy LM, không gọi proxy đó là segmentation F1. |
| [Inverse Translation](inverse-translation/IDEAS.md) | Tìm kiếm rời rạc ngược qua một model có sẵn | Phải giữ model dịch, vocab/token IDs được cấp. Đổi tokenizer BPE mới sẽ phá ID của model, nên đây là ngoại lệ bắt buộc. |
| [Alchemy](alchemy/IDEAS.md) | Cross-encoder, negative sampling và ghép cặp toàn cục | Đề cho phép BERT nhưng không bắt buộc: cả bảy lượt dùng random initialization + BPE train-only theo yêu cầu. Reference pretrained là điều kiện riêng. |
| [Tally](tally/RESULTS.md) | Weak supervision từ số span, CRF count likelihood và singleton masking | Scratch weights + train-only byte BPE; không dùng ELECTRA dù được phép. |

Mặc định BPE fit trên training split và model khởi tạo ngẫu nhiên. Riêng model/vocab do đề quy định giữ nguyên mapping: Inverse Translation dùng model dịch được cấp; Moment Match chỉ lấy fastText trong data; Alchemy không tải checkpoint, chỉ chuyển idea cross-encoder/matching sang scratch model. Không nạp checkpoint mới chỉ vì solution dùng nó.

Lời giải được dẫn nguồn là reference BTC. Chỉ gọi SOTA nếu có bằng chứng leaderboard/paper phù hợp cùng điều kiện. Điểm từ nguồn, validation tự chạy và leaderboard tự nộp luôn tách riêng. Các thay đổi là giả thuyết, không hứa chắc tăng điểm.

Sau khi lấy được output sẽ commit riêng nhật ký cho từng thí nghiệm, tối đa hai Saved Version Kaggle đang chạy. Một version có thể chứa nhiều thí nghiệm độc lập; mỗi thí nghiệm khởi tạo mới model/optimizer/seed, giữ split cũ và lưu thư mục output riêng. Số thí nghiệm GPU và số Saved Version được báo riêng. Mỗi báo cáo cần: input/output mẫu; flow baseline → reference → thay đổi; loss; cách decode; metric thật; chi phí; 3 nhóm lỗi; một câu kết luận học được gì.


Theo dõi tiến độ và kết quả thật tại [STATUS.md](STATUS.md). Mỗi thư mục `runs/Rxx/LEARNED.md` có kết luận của riêng lượt đó; không lấy điểm trong nguồn làm kết quả đã chạy.

## Tiết kiệm thao tác khi chờ GPU

Runtime đã đo: Moment Match 81–168 giây cả khởi động; Lost Interpreter R05 114 giây; Word Lookups R01 138 giây; Alchemy R01 có phần model 12.8 giây. Ước lượng batch từ tổng runtime của các thí nghiệm cộng overhead khởi động, chỉ kiểm tra khi gần mốc đó. Thời gian chờ queue được ghi riêng, không xem là thời gian train.

Dự đoán mốc kiểm tra từ dữ liệu thật: `python tools/nlp_eta.py --task word-lookups --runs 2 3 4 5 6 7`. Audit bằng chứng và số lượt duy nhất: `python tools/nlp_audit.py`. Hai lệnh này không gọi Kaggle, không cần credential. Ước lượng không bao gồm queue delay.

## Bài thứ sáu đã hoàn tất: Tally

Đã chạy và nộp đủ **7 cấu hình T4**, mỗi output có điểm và bằng chứng. R07 dẫn public **0.38386**; R06 dẫn fixed validation **0.571429** nhưng public chỉ 0.34627. Mười câu validation chưa đủ đại diện. [Bảng kết quả, lỗi và bài học](tally/RESULTS.md), [năm paper](tally/RESEARCH.md), [trạng thái](tally/STATE.json). Toàn workflow sáu bài hiện **58 cấu hình / 58 submissions có điểm**, audit pass. Không còn job Tally hoặc cấu hình tiếp theo trong batch này.
