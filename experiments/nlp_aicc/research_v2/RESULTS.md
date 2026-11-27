# Kết quả vòng research theo paper

10 cấu hình dưới đây đã train/chạy trên T4 và đã nộp Kaggle. Chỉ số trong bảng public/private là điểm Kaggle thật, hiển thị4 chữ số; không phải ước lượng từ validation. Inverse R08/R09 đã hoàn tất và được chấm; tổng toàn bộ workflow là51 cấu hình/submission có điểm.

| Bài / run | Thay đổi | Validation | Public | Private | Code T4 |
|---|---|---:|---:|---:|---:|
| Lost R09 | Typed instruction + operand values + GRU | MAE15.8136 | 14.9744 | 17.8852 | 102.09s |
| Lost R10 | R09 + mixed curriculum | MAE16.3178 | 15.2480 | 18.4884 | 84.43s |
| Alchemy R10 | Square Sinkhorn + row/column CE | 5/30 | .0285 | .1142 | 13.60s |
| Alchemy R11 | ESZSL-inspired lexical attributes | 1/30 | .0000 | .0000 | .35s |
| Moment R11 | Decomposable event cross-attention | .657224 | .6304 | .6233 | 65.00s |
| Moment R12 | Soft-DTW residual | .694051 | .6456 | .6253 | 86.56s |
| Word R08 | Accessor-variety spans | NLL5.8020 | .9150 | .9171 | 230.57s |
| Word R09 | R08 + BPE-dropout .1 | NLL5.7895 | .9145 | .9168 | 244.07s |
| Inverse R08 | Top8 + gradient fallback | Official97.2778; exact23/24 | 94.5555 | 100.0000 | 15.16s |
| Inverse R09 | Contextual gradient proposals | Official100; exact24/24 | 100.0000 | 100.0000 | 8.28s |

## Ý tưởng đáng giữ để học

**Word: hiểu đơn vị bài toán có ích hơn tăng capacity.** R07 và R08 dùng cùng LM recipe/NLL5.8020, chỉ thêm span bonus từ accessor variety. F1 tăng public .9097→.9150 (+.53 điểm phần trăm), private .9113→.9171 (+.58 điểm). Một cụm có nhiều ký tự đứng trước/sau khác nhau có dấu hiệu dùng độc lập như một từ. PMI hai ký tự chưa biểu diễn được điều đó. Đây là evidence cho thay đổi decoder; chưa phải kiểm định nhiều seed hay chứng minh mỗi span AV là từ thật.

**Proxy loss có thể đi ngược metric cuối.** BPE-dropout giảm NLL thêm nhưng F1 giảm nhẹ so R08. Không chọn model segmentation chỉ vì perplexity đẹp hơn. Hai cấu hình được chốt trước scoring; chưa sweep tham số AV/dropout theo leaderboard.

**Lost: representation rõ hơn chưa đủ để thực thi đúng.** R09 có0 lỗi parser opcode/operand ở canary, không truncate lệnh. Nó giảm MAE ở≤4 lệnh5.91→5.17, nhưng tăng ở5–8/9–16 lên17.53/23.00. MAE tổng và public/private đều kém R08. R10 đối chứng curriculum còn kém hơn. Phải giải quyết tích lũy trạng thái và độ dài, không thể suy ra rằng biến số thành float là đủ.

**Alchemy: hết UNK chưa đồng nghĩa có nghĩa.** 24/30 tên kết quả val và53/70 candidate test chưa xuất hiện ở training dưới bất kỳ vai trò nào; UNK=0. Scratch BPE ghép được chữ “phoenix”, nhưng không tự biết quan hệ vật lý/commonsense. Matching vuông đúng toán học nhưng vẫn5/30; lexical attributes ridge1/30 và test0. Không đưa BERT pretrained vào để giấu khác biệt điều kiện. Không có kết luận rằng mọi scratch architecture đều bất khả thi; chỉ có bằng chứng rằng các mô hình/feature đã thử thiếu semantic signal.

**Moment: prior có thể tăng một tập mà không tăng tập khác.** Soft-DTW tăng3/1059 câu val so R04, nhưng private .6253 thấp hơn .6284. Không tuyên bố thắng ổn định. Decomposable attention event-level bị mất local token information qua pooling là một giả thuyết cần kiểm, không phải kết luận đã chứng minh. Tăng attention/depth không tự bảo đảm reasoning.

## Thời gian và bằng chứng

**Inverse: gradient proposal phải được kiểm tra lại, không chỉ tin surrogate.** R08 bỏ lỡ row21 dù chỉ dùng3083 candidate evaluations. R09 tìm được24/24 với2600 candidates và16 gradient calls, so R06 đạt24/24 với7832 candidates: giảm66.8% số candidate được exact-evaluate. Code8.2827s so11.9711s giảm30.8% trong các lần đo này; khác batch và warm-up nên chưa coi là benchmark tốc độ lặp lại. R09 vẫn có top8 fallback; chưa có counter số lần fallback nên không gán toàn bộ gain cho gradient riêng. Đây là tín hiệu chọn bước nghiên cứu, không là bảo đảm tổng quát ngoài24 target được đề cho.

Counter `model_calls` R08/R09=90/85 là counter search (đã gồm2/16 gradient calls), không bao gồm46 probe batches mỗi run, các lần generate/canary và official evaluator. Gradient call có backward, không ngang chi phí một forward. Không dùng riêng số model_calls để khẳng định tăng tốc.

Lost batch thực tế code186.52s, Word474.64s, Moment151.56s, Alchemy13.95s. Startup/evaluator/queue tách riêng; Word Saved Version hoàn tất8m23s. Alchemy R11 là closed-form compatibility solve dùng T4, không phải một mạng deep learning đã train nhiều epoch; không cộng thành nhiều lượt từ các phép tính bên trong.

Mỗi `runs/Rxx/` có report.json, evidence.json, LEARNED.md. Bản lưu mới là các trường đo đọc từ output Kaggle, không giả mạo notebook đã execute hay full stdout. Link version chỉ rõ nguồn code immutable. Các bài có metric khác nhau, không so trực tiếp số điểm giữa bài.
