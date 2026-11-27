# Baseline → giả thuyết scratch

Đọc trực tiếp [notebook BTC](https://www.kaggle.com/code/antoningorokva/baseline-tally-aicc-round-11), version3/script355168523, ngày27/11/2026.

**Code nguồn:** pretrained ELECTRA token classifier, first-subword alignment, CE trên cả50gold,400epochs, argmax. Không đọc `train_counts.csv`; không có validation split trong code đã thấy. Không chạy lại baseline này vì Quân yêu cầu scratch.

## Thay đổi của chúng ta và phép thử

| Thành phần | R01 của chúng ta | Câu hỏi cần số đo |
|---|---|---|
| Nguồn ngôn ngữ | Byte BPE train-only và BiGRU khởi tạo ngẫu nhiên | Context từ vài nghìn câu có đủ khi thiếu pretrained không? |
| Đơn vị gán nhãn | Pool subword về word trước encoder/head | Có bảo toàn đúng nhãn cho một từ bị BPE chia nhiều mảnh không? |
| Nhãn gián tiếp | Expected run-start count loss trên weak train | Count đúng có kéo đúng vị trí thực thể không? |
| Chọn model |40/10gold split cố định, chọn checkpoint bằng span F1 | Có generalize sang câu mới và loại hiếm không? |
| Chẩn đoán |Exact span F1, lỗi biên/loại, count diagnostics riêng | Bottleneck là nhận diện loại, tìm biên hay chỉ học loss phụ? |

R01 không chỉ đổi ELECTRA thành RNN: điểm nghiên cứu là sử dụng supervision gián tiếp và đo được nó có ích hay không. Thiếu pretrained có thể tạo khoảng cách lớn; không hứa weak labels bù được khoảng cách đó.

## Vì sao chưa gọi reference là SOTA đã tái lập

Correction IOAI chưa đọc được vì trang thành viên. Các số baseline/reference trong đề là claim của nguồn; điều kiện weights và split khác R01. Listing Kaggle và viewer cũng đang chỉ các số/versions khác nhau. Chỉ ghi điểm tự chạy sau khi Kaggle chấm đúng output của phiên bất biến; không dùng điểm nguồn làm kết quả của chúng ta.

## Khi R01 hoàn tất

Đọc từng validation case và support của từng loại trước khi chọn R02. Nếu model khớp count nhưng phân bố label loãng, count likelihood đáng thử. Nếu lỗi chủ yếu là hụt biên, ưu tiên transition/span inference. Nếu loại hiếm chưa xuất hiện đủ ở40gold train, phải nêu giới hạn support; đổi loss chưa chắc xử lý được thiếu dữ liệu.
