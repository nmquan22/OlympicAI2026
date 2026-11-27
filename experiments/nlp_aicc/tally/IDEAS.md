# Tally · học vị trí thực thể từ nhãn đếm

Đây là bài thứ sáu, chọn ngày 27/11/2026 và **đã hoàn tất 7 cấu hình / 7 submissions có điểm ngày 28/11/2026**. Tổng sáu bài là 58 kết quả thật. [Báo cáo cuối](RESULTS.md): R07 dẫn public 0.38386, R06 dẫn validation 0.571429. Các đoạn tiến độ bên dưới là nhật ký lịch sử; trạng thái hiện tại ở [STATE.json](STATE.json). Không cộng kế hoạch hoặc retry.

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


## Hai phép thử tiếp đã gửi lên T4

- R02: Saved Version2/script374853542, CRF expected count từ pair marginals; [job](https://www.kaggle.com/code/shareaccc3a/aicc-nlp-tally-scratch-weak-supervision?scriptVersionId=374853542).
- R03: Saved Version3/script374853890, CRF per-type count composite likelihood; [job](https://www.kaggle.com/code/shareaccc3a/aicc-nlp-tally-scratch-weak-supervision?scriptVersionId=374853890).

Cả hai dùng nguồn bất biến `dddb8442bdc9ea57f8d474b07ad4eaf0c8cba598`, trần480s train/lượt, đang queued T4x2 lúc20:47 giờ Việt Nam. Chưa có kết quả hoặc canary pass được quan sát của R02/R03. Không cần Quân điều chỉnh từng lượt; lịch sẽ tự đọc kết quả, chọn nâng cấp trong phạm vi đã cho phép, ghi Git và thông báo khi có số mới.


## Kết quả R02/R03 (đã chấm)

Tally hiện3/7, tổng NLP54 cấu hình/scored. R02: val.444444, public.30394/private.30351,335.166s. R03: val.444444, public.35301/private.35300,374.102s. Canary cả hai pass. R03 cải thiện public và count vector nhưng ORG vẫn0/5, năm span hụt biên; chưa giải quyết được vị trí. [R03 LEARNED](runs/R03/LEARNED.md). Chuẩn bị R04 character features và R05 approximate posterior projection; chưa có kết quả của hai hướng đó.


## R04/R05 đang chạy

R04 Saved Version4/script374929474 và R05 Saved Version5/script374929750 đã RUN_START trên TeslaT4, nguồn bất biến `2127e5331fe50b6f7b102ca1943db2c55ec49612`. Canary mở rộng đã pass ở cả hai: partition/count/Viterbi/padding, KL teacher khớp brute-force enumeration, teacher stop-gradient và residual giảm trên toy. R04 đã có epoch1 với loss hữu hạn; chưa có kết quả cuối hai lượt này. Không cộng vào54 hoàn tất/scored.

## R04/R05 đã nộp và chấm, R06/R07 đã chọn

Tally5/7; tổng56 cấu hình/scored. R04: val.428571/public.37057/private.37740,330.592s/156.699MiB. R05: val.370370/public.25020/private.25609,490.202s/92.551MiB. Cả hai Complete(afterdeadline), canariespass. Count proxy tốt hơn không bảo đảm spanF1: R04 countvector38.875% nhưng val giảm, còn R05 giảm residual teacher nhưng holdout tệ. R04 hiện dẫn public; R03 giữ lựa chọn theo validation. Private chỉ báo cáo.

Hai phép thử đã chốt từ lỗi validation trước score: R06 auxiliary start/end, R07 singleton whole-word masking. Cả hai độc lập xuất phátR03, không pretrained và không thêm dữ liệu; [chi tiết giả thuyết](RESEARCH.md). Chưa tính vào56 trước khi có kết quả.

## R06/R07 đã RUN_START

Saved Version6/script374947843 và Version7/script374948086, immutable source `469e5332a09dd0faf6bcf7970cdf7c1b33d33d02`, TeslaT4x2. CRF/count/teacher canary pass cả hai; canary boundary/singleton pass R06, R07 chưa quan sát lúc snapshot cuối. Chưa có report cuối hoặc score; tổng vẫn56, Tally5/7. Notebook trongrepo là launch export chưa execute, không giả notebook đã chạy.

## Đã chốt bảy lượt

R06/R07 đã hoàn tất và nộp chính xác output; canary pass, không còn job active. R06 tăng ORG validation lên 2/5 nhưng public 0.34627; R07 tăng public lên 0.38386 dù validation thấp hơn R06. Tập 10 câu không đủ chọn chắc. Xem [RESULTS.md](RESULTS.md) cho toàn bộ bảng, baseline/reference, năm paper và các hướng chưa chạy.
