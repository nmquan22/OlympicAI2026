# Moment Match · Học idea qua thí nghiệm

**Mục tiêu học:** Late interaction và căn chỉnh chuỗi sự kiện.

## Đã đọc trước khi chạy

[Trang đề](https://www.kaggle.com/competitions/moment-match-aicc-round-10/) · [Code lời giải BTC](https://github.com/AI-Community-Contest/solutions/blob/main/round-10/moment-match.ipynb). Chưa xác minh hạng SOTA; gọi đúng là reference của BTC. Mọi số dưới đây của nguồn, chưa phải kết quả mình chạy.

**Baseline có gì:** Baseline lấy trung bình toàn bộ word vectors của mỗi cửa sổ rồi so cosine. Nó làm mất thứ tự sự kiện và làm mờ sự kiện đích.

**Lời giải mạnh phát triển ra sao:** BTC học phép chiếu fastText 300→192; so mọi cặp từ giữa mỗi cặp sự kiện, max-pool hai chiều thành 2 bảng 5×9; CNN đọc bảng và CE chọn một trong 4 ứng viên. Notebook ghi validation accuracy 0.7243. Đề ghi baseline 0.43, reference 0.70; khác bộ đo nên không đồng nhất hai số.

**Mình phát triển tiếp:** Giữ late interaction và bảng 5×9, dùng BPE fit trên train với vector khởi tạo tổng hợp từ fastText được cấp; thêm soft alignment, contextual attention và IDF. Không tải encoder pretrained.

**Weights/tokenizer:** Chỉ dùng fastText được cấp trong dữ liệu. BPE subword vectors được khởi tạo từ các word vectors này; encoder/attention/CNN học mới.

## Bảy lượt T4 đã định nghĩa

| Run | Giả thuyết cần kiểm tra | Trạng thái |
|---|---|---|
| R01 | BPE + soft late interaction | planned |
| R02 | Thêm Transformer ở mức token | planned |
| R03 | Thêm positional Transformer ở mức sự kiện | planned |
| R04 | IDF weighting cho alignment | planned |
| R05 | Giảm nhiệt độ soft alignment | planned |
| R06 | Chỉ giữ sự kiện trung tâm để đo vai trò ngữ cảnh | planned |
| R07 | Kết hợp token attention + IDF (đổi sau R03/R04) | planned |

Mỗi lượt là một Kaggle Save & Run All riêng. Tối đa 2 version chạy đồng thời. Không tính đổi seed đơn thuần là idea mới; không tính run lỗi hoặc synthetic smoke là một lượt hoàn tất. Không chạy lại baseline thấp để đủ số lượt. Sau mỗi lượt xong: ghi metric, runtime, peak VRAM, config, lỗi điển hình và câu trả lời “idea này có ích gì”; commit log để đọc ngay. Nếu một giả thuyết bị dữ liệu/evaluator bác bỏ, ghi rõ và thay bằng giả thuyết có căn cứ trước khi chạy.

**Trạng thái hiện tại:** đã đọc và phân tích nguồn; chưa có số thực nghiệm mới trong trang này.


Điều chỉnh sau kết quả: R03 event context chỉ 60.43%, R04 IDF đạt 69.12%. R07 chuyển sang token attention (R02) + IDF; in lỗi validation tự tin sai để phân tích alignment.
