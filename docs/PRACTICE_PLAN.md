# Luyện 5 tiếng để tăng độ phủ

Kế hoạch này cố ý xen bài đã quen với bài ít quen. Mỗi buổi phải tạo được prediction hợp lệ và một bảng ablation. Đây là lịch tự luyện, không phải dự báo đề.

| Buổi | Bài nên luyện | Baseline → nâng cấp cần tự thử |
|---:|---|---|
| 1 | ML bảng có group và category mới | Linear → ExtraTrees; so random/group split |
| 2 | Forecasting seasonal | Seasonal naive → ridge lag; purge target vượt cutoff |
| 3 | Audio classification | Log-mel statistics → CNN; độ dài khác nhau và SpecAugment |
| 4 | IOAI Find the Order | Directed pairs → beam; kiểm tra order/rank |
| 5 | IOAI Ghost | Char TF-IDF → encoder features; decode một changepoint |
| 6 | IOAI Robot Chasing | Flatten grid → text/robot-conditioned policy |
| 7 | SemEval 11 logic | Text classifier → entity-renaming invariance |
| 8 | DCASE 6 / EgoVis | Max clip → interval với penalty độ dài |
| 9 | PBVS EO–SAR | Single modality → masked fusion; kiểm tra missing modality |
| 10 | DataCV/noisy labels | Random subset → diversity/weights, fixed compute |
| 11 | UG2 segmentation | Direct segmenter → degradation augmentation/frontend |
| 12 | SemEval 4 narrative | Cosine → pair scorer; entity swap control |
| 13 | VAND 4.0 | Patch kNN → condition-aware memory, tách lighting shift |
| 14 | NTIRE/PBVS SR | Bicubic → residual/gated model, crop/tile đúng tọa độ |
| 15 | IOAI Potato | Nearest semantic guess → soft query selection |
| 16 | IOAI Double Agent | Single-model gradient → selective objective và norm search |
| 17 | IOAI Field rút gọn | MLP → Fourier features, loss scale, parameter budget |
| 18 | Một bài ngoài vùng quen: graph hoặc RL | GCN/label propagation hoặc Q-learning; ghi rõ phần mở rộng |

## Ngân sách 300 phút cho từng buổi

| Phút | Đầu ra cần có |
|---|---|
| 0–25 | Đọc evaluator, schema, input/target, weights được phép và đơn vị split |
| 25–65 | Baseline rẻ nhất, export rồi đọc lại prediction |
| 65–100 | Score dev; xem 10–20 lỗi, tách theo group/condition |
| 100–185 | Một nâng cấp trả lời giả thuyết lỗi; giữ cùng split/seed/data budget |
| 185–235 | Chọn cấu hình; cân nhắc ensemble chỉ khi dev cho thấy bổ sung nhau |
| 235–270 | Chạy inference đầy đủ, đo runtime, khóa config |
| 270–300 | Fresh-process export; kiểm tra IDs, shape, class order, offsets/ranks |

Nếu một epoch tốn phần lớn buổi, giảm resolution/context/subset hoặc cache frozen features. Tính thời gian load/model initialization và export, không chỉ GPU forward. T4 và H100 MIG có profile khác nhau; đo trên thiết bị mình thực sự dùng.

## Những giả định nên chủ động phá

| Giả định dễ mắc | Control để kiểm tra |
|---|---|
| Transformer luôn hơn ML cổ điển | So với TF-IDF/ridge/tree trên cùng dev |
| Có audio thì phải chạy ASR | So log-mel/embedding trực tiếp với ASR-based representation |
| Pooling toàn ảnh/video là đủ | Shuffle regions/frames; thử head không gian/thời gian |
| Metric trung bình tốt thì generalize tốt | Worst-group, condition split, held-out scene/subject |
| Thêm RGB luôn giúp thermal | Che hoặc dịch RGB ở dev; so thermal-only |
| Hai kiến trúc khác nhau sẽ ensemble tốt | Đo disagreement, lỗi giao nhau và blend score thực |
| Mọi đề vision đều cần pretrained backbone lớn | Thử coordinate MLP, prototype và statistical controls |
| Dữ liệu test không bao giờ có label | Đọc luật cụ thể: Double Agent Dilemma cho labels để tối ưu perturbation; nhiều bài khác không cho |

Khi ghi kết quả, dùng [mẫu log](EXPERIMENT_TEMPLATE.csv). Luôn ghi `synthetic`, `real_subset` hoặc `official_full` đúng phạm vi; không đổi tên smoke test thành benchmark.
