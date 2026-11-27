# Tally · học vị trí thực thể từ nhãn đếm

Đây là bài thứ sáu, được chọn theo yêu cầu ngày 27/11/2026. **R01 đã hoàn tất và nộp:** valF1=0.444444 (10câu), Kaggle public0.29710/private0.30080; chưa đạt mục tiêu bảy. Năm bài trước có51 cấu hình; tổng hiện52 kết quả thật. Không cộng kế hoạch hoặc retry.

## Vì sao bài này đáng học

[Đề Kaggle](https://www.kaggle.com/competitions/tally-aicc-round-11/overview) cho 8.000 câu với số tên thuộc từng loại và 50 câu có nhãn từng từ. Test có 4.000 câu. Đích là tìm đúng **start, end, type** của mỗi tên; điểm là micro entity-span F1.

Một chuỗi `PERSON PERSON O PERSON` có hai tên PERSON, không phải ba. Nhiều chuỗi khác nhau có cùng số tên, vì thế loss đếm nhỏ chưa chứng minh model tìm đúng vị trí. Tên giả hiếm còn làm cách học thuộc từ mất tác dụng. Đây là cơ hội học weak supervision và structured prediction, khác năm bài đã làm.

Đề cho phép ELECTRA nhưng không bắt buộc. Theo điều kiện của Quân, toàn bộ weights khởi tạo ngẫu nhiên, BPE byte-level fit trên train. Không dùng tokenizer ELECTRA, dữ liệu ngoài, nhãn test hoặc nhãn do LLM viết. Dấu câu có thể nằm **bên trong** tên; quy tắc không được bắt đầu/kết thúc bằng dấu câu không đồng nghĩa cấm mọi dấu câu trong span.

## Lượt đầu đã chạy và nộp

`text.split()` → BPE cho từng từ → mean-pool các subword của từ → ghép log tần suất/độ dài từ → packed BiGRU hai lớp → head sáu nhãn IO.

- Gold CE định vị/đặt tên năm loại; replay 16 câu train mỗi weak batch.
- Weak loss khớp kỳ vọng số lần bắt đầu mỗi loại, không khớp tổng xác suất token.
- Với phân phối nhãn độc lập: `E[count_c] = sum_t p_t(c) * (1 - p_(t-1)(c))`, đặt xác suất trước từ đầu là 0. Công thức chính xác trong giả định độc lập này; không gọi đây là pair marginal CRF.
- Mỗi batch weak có 64 câu, warm-up 160 bước gold, tối đa 35 epoch/480 giây phần train, patience 8. Dừng có thể vượt ngân sách một bước/evaluation nhỏ.
- Tách cố định 40 gold train / 10 gold validation và 7.200 weak train / 800 weak holdout, seed42. Loại weak train trùng text gold validation. BPE/tần suất chỉ fit tập train. Không cắt ngắn câu.
- Chọn checkpoint bằng validation F1. Dùng đúng checkpoint đó để submit, chưa refit cả 50 gold. Mười câu chỉ là pilot nhỏ; không tuyên bố ý nghĩa thống kê hay SOTA.

Nguồn chạy: [tally.py](../code/tally.py), qua `run_batch('tally', [1], SOURCE_COMMIT)` và commit SHA bất biến. Notebook xuất ra chưa execute phải giữ nhãn đó.

## Vòng nâng cấp sau khi đọc lỗi thật

R01 đã có bằng chứng. R02/R03 đã chọn theo lỗi đo được: CRF expected counts và CRF count likelihood; xem [kết quả R01](runs/R01/LEARNED.md). Các nhánh khác dưới đây vẫn chỉ là kế hoạch, không tính là kết quả.

| Lỗi quan sát cần có | Câu hỏi | Nâng cấp phù hợp để thử |
|---|---|---|
| Count MAE giảm nhưng F1 yếu, xác suất loãng | Moment đúng có đủ không? | Likelihood của count qua DP; so với matching expectation |
| Tách/gộp tên nhiều | Các nhãn láng giềng có cần phụ thuộc? | CRF và pair marginals; cùng encoder, cùng split |
| Biên sai ở article/dấu câu/possessive | Có vi phạm tri thức đề không? | Constrained decoding chỉ ở start/end, không cấm punctuation nội bộ |
| Tên lạ kém hơn từ gặp ở train | BPE có thiếu thông tin hình thái? | Character encoder hoặc dropout ở subword, không thêm từ điển ngoài |
| Span dài có biên sai | Head token có lệch metric không? | Semi-Markov/span head với độ dài và biên |
| Pseudo-label tự củng cố lỗi | Có thể dùng count để lọc/project teacher không? | Posterior projection + KL student, chỉ trên weak train |

Tối thiểu bảy cấu hình có ý tưởng khác nhau sẽ được chọn dần từ kết quả, không đếm retry, chỉ đổi seed, hay Saved Version thành cấu hình mới. Tối đa hai Saved Version đồng thời; lượt đầu chỉ cần một vì vòng sau phụ thuộc lỗi lượt đầu.

## Bằng chứng và thời gian

Mỗi report lưu F1 exact span, TP/FP/FN, F1/support từng loại, lỗi đúng biên sai loại, đúng loại sai biên, bỏ sót, all-O, vi phạm quy tắc biên, recall entity chứa từ chưa gặp. Weak holdout lưu count MAE và tỷ lệ đúng cả vector riêng; đó không phải entity F1. Output chứa prediction/truth của cả mười câu validation để đọc lỗi.

Trước khi có số T4, 480 giây là **trần train cấu hình**, không phải ETA đo được. Kiểm lần đầu khoảng 12–15 phút sau khi job bắt đầu chạy để chừa khởi động/encoding/inference. Queue delay ghi riêng; nếu queued, dùng backoff 15→30→60 phút, không mở liên tục. Khi có runtime thật, dùng thời gian đó + buffer startup để đặt mốc kiểm tiếp.

## Nguồn lời giải và giới hạn hiện tại

- [Baseline BTC trên Kaggle](https://www.kaggle.com/code/antoningorokva/baseline-tally-aicc-round-11). Trang Code hiển thị score 0.68172; phần đề ghi baseline 0.66/reference 0.83. Đây là số **của nguồn**, chưa xác minh cùng split/weights với scratch và không phải kết quả của chúng ta. Đã đọc source version3 (script355168523): xem [phân tích baseline](BASELINE_REVIEW.md). Viewer đồng thời ghi best score0.67398 ởV2; chưa xác minh vì sao khác listing, không dùng hai số này để kết luận cải thiện.
- [Correction IOAI](https://ioai.artix.tech/problems/aicc-2026-r11-tally/correction) đang yêu cầu tài khoản thành viên. Chưa đọc được nội dung; không gọi đó là SOTA đã tái lập.
- [Năm paper và cách chuyển ý tưởng](RESEARCH.md). Chỉ chọn triển khai sau khi đối chiếu lỗi; đọc paper không đồng nghĩa đã chạy phương pháp đó.
- Đã chấp nhận quy định riêng Tally sau xác nhận “đồng ý hết giúp mình” ngày 27/11/2026 và quan sát thông báo thành công của Kaggle. [R01 Saved Version 1](https://www.kaggle.com/code/shareaccc3a/aicc-nlp-tally-scratch-weak-supervision?scriptVersionId=374844445), nguồn `08c63dcfb7af4f7c08c21bf0487e78dee5ee9585`, đã chạy xong và nộp Complete(afterdeadline). Không tính queue/retry là cấu hình mới.
