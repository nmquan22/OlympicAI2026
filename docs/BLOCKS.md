# Bản đồ khối code để chọn nhanh

Các model mới mặc định from-scratch, không truy cập mạng. Khối dùng cached embedding nhận feature do mình tạo từ weights được phép; chúng không tự mang theo CLIP/BGE/Whisper. Bảng này tách chương trình chạy trọn luồng và thành phần cần adapter.

| Thể loại | Khối mới | Đầu vào → đầu ra | Trạng thái |
|---|---|---|---|
| ML bảng | [tabular](../ml/blocks/tabular.py), [runner](../ml/tabular.py) | CSV numeric/category → prediction/model | Train, evaluate, export; classification/regression |
| Validation | [split + group metric](../ml/blocks/validation.py) | groups/time → indices, scores | Thành phần; chọn split theo data generating process |
| Forecasting | [windows/baselines](../ml/blocks/timeseries.py), [runner](../ml/forecast.py) | NPY `[T,C]` → forecast/MAE | Trọn luồng cho regular series không NaN |
| Sensor/video sequence | [TemporalCNN](../ml/blocks/timeseries.py) | `[B,T,C]` + lengths → logits | Thành phần; padding-aware |
| Graph | [GCN/label propagation](../ml/blocks/graphs.py) | graph nhỏ + train labels → node logits | Thành phần, dense adjacency O(N²) |
| Behavior cloning | [InstructionPolicy](../ml/blocks/behavior.py) | categorical grid + instruction → action | Thành phần theo schema IOAI Robot Chasing |
| Interactive search | [ComparisonSearch](../ml/blocks/active_search.py) | word features + feedback → candidate | Heuristic; cần judge/protocol adapter |
| RL 2027 | [RL losses](../ml/blocks/rl.py), [chain demo](../ml/q_learning.py) | transitions/episodes → updates | Q-learning demo; DQN/REINFORCE là khối loss |
| Text retrieval | [sparse + RRF + nDCG](../nlp/blocks/retrieval.py) | corpus/query → ranks | Thành phần; dense search cần encoder được cấp |
| Dialogue order | [PairScorer + beam](../nlp/blocks/ordering.py), [decoder CLI](../nlp/order.py) | directed scores → ranks JSON | Decode/export; feature training cần adapter |
| Change point | [offsets + decoder](../nlp/blocks/changepoint.py), [runner](../nlp/boundary.py) | labelled JSONL → boundary JSONL | Trọn luồng TF-IDF CPU; chưa fine-tune BGE |
| Token/span output | [structured](../nlp/blocks/structured.py) | logits/offset mapping → span/loss | Thành phần; phải align bằng tokenizer thật |
| Restoration/SR | [residual + GuidedSR + tile](../vision/blocks/restoration.py), [runner](../vision/restore.py) | image pairs → restored image | Runner RGB pairs; GuidedSR là block thermal riêng |
| Robustness | [selective perturbation](../vision/blocks/robustness.py) | hai classifiers + raw image → delta | Thành phần; cần official preprocess/export/scorer |
| Self-supervision | [NT-Xent + EMA](../vision/blocks/self_supervised.py) | hai views → loss/teacher update | Thành phần; batch >=2 source images |
| Implicit field | [FourierField](../vision/blocks/fields.py) | coordinates → value | Thành phần; chưa có stochastic IOAI Field solver |
| Missing modalities | [MaskedFusion](../vision_ml/blocks/multimodal.py) | vectors + availability → logits | Thành phần; mỗi sample phải có >=1 modality |
| Few-shot | [ridge prototypes](../vision_ml/blocks/multimodal.py) | support/query features → logits | Thành phần; dual solve theo số support |
| Audio frontend | [WAV/log-mel/SpecAugment](../audio/blocks/features.py) | WAV → features | Mono-only loader; multi-channel cần adapter |
| Audio classification | [CNN + runner](../audio/classify.py) | WAV CSV → prediction/checkpoint | Trọn luồng single-label; cache RAM |
| CTC + WER primitives | [audio models](../audio/blocks/models.py) | frame logits → tokens; sequences → edit count | Thành phần, không có ASR training recipe |
| Temporal grounding | [interval decoder](../audio/blocks/temporal.py) | clip scores/time → interval | Dùng chung audio/video; chưa official-format adapter |
| Distillation/budget | [runtime](../common/runtime.py) | logits/timer → loss/time budget | Thành phần; teacher class order phải khớp |
| Ensemble/calibration | [ensemble](../common/ensemble.py) | aligned logits/probs → calibrated blend | Fit trên dev/calibration, freeze trước test |
| Submission | [export/IDs/ranks](../common/submission.py) | IDs/values → CSV/JSON | Chặn duplicate ID và permutation sai |

## Các khối gốc vẫn dùng được

- [NLP](../nlp/README.md): data/BPE, Transformer encoder, decoder-only causal, CoNLL readers, TF-IDF baseline.
- [Vision](../vision/README.md): bbox COCO/VOC/YOLO, detection, segmentation, masks, tracking.
- [Vision + ML](../vision_ml/README.md): patch anomaly, image–text contrastive, spatial question head, prototype và retrieval nhiều positive.

## Chọn ensemble bằng evidence

Hai model phải xuất dự đoán trên cùng sample IDs, class order và split. Dùng `disagreement` để tìm mẫu bổ sung nhau; kiểm tra blend thật trên dev thay vì kết luận từ kiến trúc khác nhau. Đo tổng inference time trước khi thêm model. Temperature dùng calibration split; nếu vừa tune weights vừa báo điểm cùng split thì ghi rõ bias chọn cấu hình, hoặc giữ thêm một holdout.
