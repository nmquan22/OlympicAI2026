# Lost Interpreter · Học idea qua thí nghiệm

**Mục tiêu học:** Biểu diễn chương trình, số và luồng điều khiển.

## Đã đọc trước khi chạy

[Trang đề](https://www.kaggle.com/competitions/lost-interpreter-aicc-round-5/) · [Code lời giải BTC](https://github.com/AI-Community-Contest/solutions/blob/main/round-5/lost-interpreter.ipynb). Chưa xác minh hạng SOTA; gọi đúng là reference của BTC. Mọi số dưới đây của nguồn, chưa phải kết quả mình chạy.

**Baseline có gì:** Lời giải được công bố tự mô tả là LSTM nhưng code là BiGRU, pooling mean/max và MLP hồi quy, SmoothL1 trên target chuẩn hóa. Vocabulary trong nguồn lấy cả train và test; code mới sẽ fit BPE chỉ trên phần train.

**Lời giải mạnh phát triển ra sao:** Nguồn BTC ghi validation MAE 22.6452 và có cell đọc solution.csv sau đó. Đây không phải bằng chứng SOTA; không dùng nhãn test từ cell đó. Đề chỉ cho biết mỗi chương trình là chuỗi lệnh, có thể có jump/control flow và không được dùng pretrained.

**Mình phát triển tiếp:** Thay GRU bằng Transformer encoder khởi tạo ngẫu nhiên, BPE, head hồi quy tối ưu MAE. Tách thí nghiệm đánh dấu lệnh, biểu diễn số, query pooling và auxiliary loss. Không giả định Transformer chắc chắn tốt hơn.

**Weights/tokenizer:** Không dùng bất kỳ pretrained weights nào.

## Bảy lượt T4 đã định nghĩa

| Run | Giả thuyết cần kiểm tra | Trạng thái |
|---|---|---|
| R01 | BPE Transformer + mean/max + MAE | planned |
| R02 | Thêm separator lệnh và marker số | planned |
| R03 | Nhánh thống kê số fit trên train | planned |
| R04 | Learned query attention pooling | planned |
| R05 | Auxiliary loss dự đoán số lệnh | planned |
| R06 | Signed-log target để kiểm tra outlier | planned |
| R07 | Kết hợp structure + numeric + query + auxiliary | planned |

Mỗi lượt là một Kaggle Save & Run All riêng. Tối đa 2 version chạy đồng thời. Không tính đổi seed đơn thuần là idea mới; không tính run lỗi hoặc synthetic smoke là một lượt hoàn tất. Không chạy lại baseline thấp để đủ số lượt. Sau mỗi lượt xong: ghi metric, runtime, peak VRAM, config, lỗi điển hình và câu trả lời “idea này có ích gì”; commit log để đọc ngay. Nếu một giả thuyết bị dữ liệu/evaluator bác bỏ, ghi rõ và thay bằng giả thuyết có căn cứ trước khi chạy.

**Trạng thái hiện tại:** đã đọc và phân tích nguồn; chưa có số thực nghiệm mới trong trang này.

