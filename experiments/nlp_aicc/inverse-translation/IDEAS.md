# Inverse Translation · Học idea qua thí nghiệm

**Mục tiêu học:** Tìm kiếm rời rạc ngược qua một model có sẵn.

## Đã đọc trước khi chạy

[Trang đề](https://www.kaggle.com/competitions/inverse-translation-aicc-round-10/) · [Code lời giải BTC](https://github.com/AI-Community-Contest/solutions/blob/main/round-10/inverse-translation.ipynb). Chưa xác minh hạng SOTA; gọi đúng là reference của BTC. Mọi số dưới đây của nguồn, chưa phải kết quả mình chạy.

**Baseline có gì:** Baseline đề xuất token Alien ngẫu nhiên và chấm mỗi ứng viên bằng edit distance của chuỗi sinh ra. Cách này tốn nhiều autoregressive decoding và nhận tín hiệu tối ưu rất thưa. Không chạy lại baseline này.

**Lời giải mạnh phát triển ra sao:** BTC probe từng token Alien, giữ top-32 token đầu vào cho mỗi token English, rồi coordinate search theo margin max-logit trừ target-logit. Hook vị trí decoder khiến teacher-forcing phù hợp với cache của generate; evaluator gốc vẫn là chuẩn cuối cùng. Notebook có 24/24 exact match. Đề ghi baseline 7.23, reference 100.

**Mình phát triển tiếp:** Vì reference đã đạt trần trên 24 targets, phát triển theo hiệu quả bộ nhớ/tốc độ và độ ổn định, không bịa thêm tăng accuracy. Probe chia batch nhỏ, top-k streaming và xác minh bằng evaluator gốc.

**Weights/tokenizer:** Phải giữ model dịch, vocab/token IDs được cấp. Đổi tokenizer BPE mới sẽ phá ID của model, nên đây là ngoại lệ bắt buộc.

## Bảy lượt T4 đã định nghĩa

| Run | Giả thuyết cần kiểm tra | Trạng thái |
|---|---|---|
| R01 | Probe streaming giữ top-32, margin search | planned |
| R02 | Top-8 để đo recall–cost | planned |
| R03 | Top-16 để đo recall–cost | planned |
| R04 | Cross-entropy thay margin, cùng candidate budget | planned |
| R05 | Coarse-to-fine top-8→32 | planned |
| R06 | Coordinate sweep thay full replacement grid | planned |
| R07 | Restart chỉ cho case chưa exact, đo tổng chi phí | planned |

Mỗi lượt là một Kaggle Save & Run All riêng. Tối đa 2 version chạy đồng thời. Không tính đổi seed đơn thuần là idea mới; không tính run lỗi hoặc synthetic smoke là một lượt hoàn tất. Không chạy lại baseline thấp để đủ số lượt. Sau mỗi lượt xong: ghi metric, runtime, peak VRAM, config, lỗi điển hình và câu trả lời “idea này có ích gì”; commit log để đọc ngay. Nếu một giả thuyết bị dữ liệu/evaluator bác bỏ, ghi rõ và thay bằng giả thuyết có căn cứ trước khi chạy.

**Trạng thái hiện tại:** đã đọc và phân tích nguồn; chưa có số thực nghiệm mới trong trang này.

