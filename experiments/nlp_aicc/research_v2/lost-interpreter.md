# Lost Interpreter: học biểu diễn phép thực thi

## Bài khó ở đâu?

Input là chuỗi OP1..OP12 cùng số nguyên; target là một số, không có trace của trạng thái trung gian. RNN trên BPE giảm MAE rất mạnh so với Transformer nhỏ. Điều này ủng hộ thử bias tuần tự, chưa chứng minh GRU tự học đúng semantics. R08 sai trung bình 5.91 ở ≤4 lệnh, 16.64 ở 5–8, 22.01 ở 9–16. Pooling token có thể làm mờ lệnh, thứ tự và vai trò operand. Parser numeric cũ còn đọc nhầm chữ số opcode; R03/R07 không phải bằng chứng sạch rằng số học không có ích. R08 không dùng nhánh ấy.

## 5 paper ưu tiên

| Hạng | Paper / nguồn gốc | Ý chính dùng được | Điều kiện không được bỏ qua | Quyết định |
|---|---|---|---|---|
| 1 | [Learning to Execute, Zaremba & Sutskever, 2015](https://arxiv.org/html/1410.4615v3) §3–4, §8 | Recurrent execution; curriculum kết hợp ví dụ dễ với phân phối đầy đủ | Chương trình và supervision khác; paper lưu ý teacher forcing; không đem accuracy digit so MAE | R09 biểu diễn instruction, R10 curriculum hỗn hợp |
| 2 | [Neural Arithmetic Units, Madsen & Johansen, ICLR 2020](https://arxiv.org/pdf/2001.05016) §1–3 | Tách độ lớn số khỏi ký hiệu; phép cộng/nhân cần bias phù hợp, kiểm soát ổn định | NMU không có phép chia; gates khó học; không đủ biết OP là phép gì | R09 đưa operand đúng giá trị. Chưa gọi nhánh này NAU/NMU |
| 3 | [Neural Programmer-Interpreters, Reed & de Freitas, ICLR 2016](https://arxiv.org/html/1511.06279v4) §3 | Tách mã chương trình, arguments, recurrent core | NPI dùng fully supervised execution traces và program memory; bài này chỉ có output cuối | Mượn typed input; không tuyên bố reproduce NPI |
| 4 | [Neural GPUs Learn Algorithms, Kaiser & Sutskever, ICLR 2016](https://arxiv.org/html/1511.08228v3) §2–3 | Repeated shared transitions, kiểm tra khả năng tổng quát theo độ dài | Binary arithmetic và training recipe chuyên biệt; không tự suy ra control flow OP | Chưa chạy Neural GPU nguyên bản; ưu tiên instruction sequence nhỏ hơn |
| 5 | [Learning Simple Algorithms from Examples, Zaremba et al., ICML 2016](https://proceedings.mlr.press/v48/zaremba16.html) | Controller với tape/grid, tách dữ liệu khỏi thao tác | Q-learning + interface khác, cần thiết kế môi trường; quá nhiều biến thay đổi | Để dành nếu lỗi đòi memory/pointer; không thêm RL vào lượt hiện tại |

## Thí nghiệm đã định trước

**R09:** parse `OP11 7 10` thành opcode 11 và operands [7,10], không thành các số 1,7,10. Mỗi instruction: mean embedding BPE64 + opcode32 + từng operand theo vị trí (value/20, signed-log, presence mask) → MLP128 → GRU hai lớp một chiều → masked mean/max + trạng thái cuối → scalar. Không cắt instruction/operand; kích thước padding đo từ cấu trúc input. Train-only BPE2048; SmoothL1 beta .2; holdout/seed giống R08; tối đa 60 epoch/300 giây.

**R10:** giữ nguyên R09, chỉ đổi sampling trong 20 epoch đầu: nửa từ nhóm dễ tăng dần theo số lệnh, nửa từ toàn bộ training. Sau đó toàn bộ phân phối. Không sinh pseudo program/target khi chưa biết interpreter.

R09 đổi granularity và hướng recurrence cùng lúc: nếu tăng không được gán hết cho numeric channel. R10 so với R09 mới là ablation riêng curriculum. Báo MAE theo độ dài, lỗi lớn nhất, thời gian và VRAM. Chưa có kết quả GPU vòng 2 tại thời điểm viết kế hoạch này.
