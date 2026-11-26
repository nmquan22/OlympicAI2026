# Word Lookups · Học idea qua thí nghiệm

**Mục tiêu học:** Surprisal làm tín hiệu ranh giới và giải mã BMES.

## Đã đọc trước khi chạy

[Trang đề](https://www.kaggle.com/competitions/aicc-round-9-word-lookups/) · [Code lời giải BTC](https://github.com/AI-Community-Contest/solutions/blob/main/round-9/word-lookups.ipynb). Chưa xác minh hạng SOTA; gọi đúng là reference của BTC. Mọi số dưới đây của nguồn, chưa phải kết quả mình chạy.

**Baseline có gì:** Không có baseline thấp được chạy lại. Ý tưởng trong nguồn: train hai language model một chiều theo ký tự, rồi dùng độ bất ngờ ở hai hướng để đặt ranh giới từ.

**Lời giải mạnh phát triển ra sao:** Code BTC dùng forward/backward LSTM riêng, cộng surprisal rồi threshold thành words/BMES. Có các điểm cần sửa trước khi tin kết quả: flip cả tensor đã padding; comment nói vocab train-only nhưng code ghép test; có cell dùng solution.csv; hàm boundary_f1 ở phần minh họa nhận gold BMES như danh sách từ. Không dùng các cell gold này làm evaluator.

**Mình phát triển tiếp:** Causal Transformer LM khởi tạo mới, BPE với offset ký tự và decoder ranh giới hợp lệ. So sánh surprisal, coherence thống kê từ train, và ràng buộc độ dài bằng DP. BPE boundary không mặc định là word boundary; phải có phép chiếu và kiểm tra offset.

**Weights/tokenizer:** Không pretrained, external dictionary hoặc nhãn thủ công. Giữ test chỉ để inference; tách train holdout làm proxy LM, không gọi proxy đó là segmentation F1.

## Bảy lượt T4 đã định nghĩa

| Run | Giả thuyết cần kiểm tra | Trạng thái |
|---|---|---|
| R01 | Transformer surprisal với offset-safe BPE | planned |
| R02 | Surprisal hai chiều | planned |
| R03 | Chuẩn hóa surprisal theo vị trí/độ dài | planned |
| R04 | Kết hợp cohesion n-gram học từ train | planned |
| R05 | DP với prior độ dài từ | planned |
| R06 | Punctuation-aware BMES constraints | planned |
| R07 | Kết hợp hai chiều + cohesion + DP | planned |

Mỗi lượt là một Kaggle Save & Run All riêng. Tối đa 2 version chạy đồng thời. Không tính đổi seed đơn thuần là idea mới; không tính run lỗi hoặc synthetic smoke là một lượt hoàn tất. Không chạy lại baseline thấp để đủ số lượt. Sau mỗi lượt xong: ghi metric, runtime, peak VRAM, config, lỗi điển hình và câu trả lời “idea này có ích gì”; commit log để đọc ngay. Nếu một giả thuyết bị dữ liệu/evaluator bác bỏ, ghi rõ và thay bằng giả thuyết có căn cứ trước khi chạy.

**Trạng thái hiện tại:** đã đọc và phân tích nguồn; chưa có số thực nghiệm mới trong trang này.

