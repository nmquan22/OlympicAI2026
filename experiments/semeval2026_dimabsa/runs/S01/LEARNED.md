# S01 · Từ hồi quy sang trích xuất liên kết

## Kết quả thực chạy

- Kaggle T4, torch 2.11.0+cu130, 1.292.019 parameter, 430 giây train. Epoch được chọn: 30/30; best còn ở cuối ngân sách nên chưa có bằng chứng model đã hội tụ.
- Official dev: ST2 cF1 **0.588162**, ST3 cF1 **0.547711**. Scorer BTC và local metric khớp tới sai số số thực.
- 1.447 câu train chứa 2.374 quan hệ span sau gộp. Nhiều annotation cùng aspect–opinion được gộp VA, category lấy annotation đầu. Policy này có thể mất tín hiệu category; cần kiểm tra trước khi nâng model.
- 486 tuple dự đoán, 408 gold; đúng 287 pair, đúng cả category 267. Ngưỡng span/link đã chọn trên dev: **0.7/0.7**. Chưa chấm test.

## Nút thắt nằm ở đâu?

1. Top-8 aspect/opinion candidate chứa **369/408 = 90,44%** pair gold. Sau threshold/link còn **287/408 = 70,34%**. Có nhiều cặp bị loại hoặc ghép sai ở bước cuối. 90,44% là upper bound recall ở bộ candidate hiện tại, không phải F1 model tự đạt.
2. Pair precision **59,05%**, recall **70,34%**, F1 chưa tính VA **64,21%**. Span aspect riêng F1 **82,48%**, opinion riêng **80,11%**: có span đúng vẫn chưa đủ ghép đúng quan hệ.
3. Khi pair đúng, category accuracy **93,03%**, RMSE_VA **1,1702** trên 287 pair matched. Đây là metric có điều kiện, không so trực tiếp với ST1 trên toàn bộ aspect.
4. Flat category head có 14 class train; dev có 6 tuple thuộc 3 tổ hợp category chưa gặp trong train. Không thêm lớp theo gold dev. Ý tưởng phiên sau: factorize entity/attribute theo schema đề để generalize tổ hợp; ưu tiên pair precision/recall vì đó là nguồn mất điểm lớn hơn.

## Cấu trúc cần nhớ trong phòng thi

`BPE random → mean subpiece từng word → BiLSTM → enumerate spans → A/O heads → candidate pairs → link + VA + category`

- Span [s,e]: ghép h_s, h_e, mean h_s…h_e, embedding độ dài. Prefix sum tính mean span.
- Pair (a,o): ghép z_a, z_o, z_a*z_o, abs(z_a-z_o), embedding khoảng cách; MLP chung rồi các head.
- Train dùng cả gold span và span model đang dự đoán để pair head gặp hard negative. Nếu chỉ feed gold span thì inference khác train.
- Loss: balanced BCE span/link, MSE VA có scale, CE category. Lấy mẫu/balanced BCE không tự đảm bảo xác suất đúng prevalence; chọn threshold bằng dev.
- Đây là một encoder/seed; chưa có ablation để quy mức cải thiện riêng cho hard negative hay span representation.

## Bước tiếp theo đã khóa

E01 chỉ inference: S01 v2, ngưỡng 0.7/0.7 trên test; ST1 0.5 C02 + 0.5 R01. Không quét decoder/model bằng điểm test. Workspace/browser bị mất kết nối trước khi xác minh upload input mới hoặc gửi E01; chưa có điểm test.

Lỗi preprocessing S01 v1 ở ../../failures/S01_v1.md, tách khỏi thí nghiệm train hoàn tất. Full log, predictions và checkpoint giữ ở [Kaggle Output](https://www.kaggle.com/code/quannguyen0501/dimabsa-s01-scratch-joint-span-pair-va/output?scriptVersionId=374897225). Các report/diagnostics trong git được giữ từ dữ liệu đã đọc trước lúc mất môi trường.
