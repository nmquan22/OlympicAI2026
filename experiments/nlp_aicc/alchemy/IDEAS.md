# Alchemy · Học idea qua thí nghiệm

**Mục tiêu học:** Cross-encoder, negative sampling và ghép cặp toàn cục.

## Đã đọc trước khi chạy

[Trang đề](https://www.kaggle.com/competitions/alchemy-aicc-round-4/) · [Code lời giải BTC](https://github.com/AI-Community-Contest/solutions/blob/main/round-4/alchemy.ipynb). Chưa xác minh hạng SOTA; gọi đúng là reference của BTC. Mọi số dưới đây của nguồn, chưa phải kết quả mình chạy.

**Baseline có gì:** Baseline encode riêng cặp item và candidate bằng BERT CLS, cosine similarity, rồi greedy chọn candidate chưa dùng. Greedy phụ thuộc thứ tự dòng và có thể chặn một ghép tốt hơn về sau. Đề ghi baseline khoảng 0.08.

**Lời giải mạnh phát triển ra sao:** BTC chấm trực tiếp chuỗi combining a and b creates r bằng BERT cross-encoder; train CE trên một pool kết quả, augment hoán đổi a/b, trung bình score hai chiều và dùng Hungarian assignment. Đề ghi reference khoảng 0.41; notebook code không có leaderboard score nên không gán một score khác cho nó.

**Mình phát triển tiếp:** Giữ candidate-aware scoring và matching toàn cục. Thử negative khó, symmetry consistency, loss cho toàn candidate list, độ sâu scratch encoder và Sinkhorn. Tách việc model chấm điểm tốt hơn khỏi việc decoder tận dụng ràng buộc một-một.

**Weights/tokenizer:** Đề cho phép bert-base-uncased nhưng không bắt buộc. Theo yêu cầu người học, cả bảy lượt dùng random initialization và BPE fit trên train fold. Chỉ chuyển idea của reference; không coi điểm pretrained là đối chứng ngang điều kiện.

## Bảy lượt T4 đã định nghĩa

| Run | Giả thuyết cần kiểm tra | Trạng thái |
|---|---|---|
| R01 | Cross-encoder + symmetry consistency | planned |
| R02 | Hard negative mining từ train | planned |
| R03 | Candidate-list CE thay sampled negatives | planned |
| R04 | Bốn scratch encoder layers thay hai | planned |
| R05 | Trung bình xác suất hai thứ tự thay trung bình logits | planned |
| R06 | Sinkhorn training + đối chiếu Hungarian/greedy/independent | planned |
| R07 | Kết hợp hard negatives + full-list CE + probability averaging | planned |

Mỗi lượt là một Kaggle Save & Run All riêng. Tối đa 2 version chạy đồng thời. Không tính đổi seed đơn thuần là idea mới; không tính run lỗi hoặc synthetic smoke là một lượt hoàn tất. Không chạy lại baseline thấp để đủ số lượt. Sau mỗi lượt xong: ghi metric, runtime, peak VRAM, config, lỗi điển hình và câu trả lời “idea này có ích gì”; commit log để đọc ngay. Nếu một giả thuyết bị dữ liệu/evaluator bác bỏ, ghi rõ và thay bằng giả thuyết có căn cứ trước khi chạy.

**Trạng thái hiện tại:** đã đọc và phân tích nguồn; chưa có số thực nghiệm mới trong trang này.


Temperature chung hoặc trừ một hằng số mỗi hàng không đổi nghiệm Hungarian. Vì vậy R05 kiểm tra trung bình **xác suất hai thứ tự**, một phép biến đổi thực sự có thể đổi assignment. R06 thêm loss toàn cục, không chỉ chạy lại cùng model để đủ số lượt.
