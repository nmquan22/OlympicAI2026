# CV + ML và đa phương thức · 30 bài luyện

Bắt đầu từ một ảnh tạo ra một vector, một lưới patch hoặc một tập vector vùng. Khi đã cache đúng đơn vị, có thể thử kNN, prototypes, ridge, dual encoder và cross-attention trên CPU hoặc một GPU nhỏ. Với video, cache thêm frame index hoặc timestamp; với ảnh–text, giữ image ID cho từng caption.

## Học code theo thứ tự

| Khối | Đầu vào → đầu ra | Điều cần đọc |
|---|---|---|
| [Patch memory](blocks/anomaly.py) · [chạy anomaly](anomaly.py) | normal feature grid → image score + heatmap | Fit scaler và kNN chỉ bằng normal train; reshape điểm về đúng lưới patch |
| [Prototype và Recall@K](blocks/metrics.py) · [chạy Omniglot](oneshot.py) | support/query → nhãn hoặc rank | L2 normalize, nhiều positives và episodic sampling |
| [Dual encoder và spatial attention](blocks/fusion.py) | cached vectors → ma trận similarity; grid + câu hỏi → answer logits | Normalize phép chiếu, contrastive loss hai chiều, attention có điều kiện theo câu |
| [Train và đánh giá retrieval](retrieve.py) | NPZ ảnh–caption → R@K hai chiều | Một ảnh có thể có nhiều caption. Gallery ảnh lấy một vector cho mỗi image ID |
| [Metadata classifier](blocks/tabular.py) | cột số và phân loại → class logits/probabilities | Impute, standardize, one-hot và logistic trong một sklearn pipeline |

Ví dụ, với image IDs `[7, 7, 12]` và ba caption tương ứng, hai caption đầu đều đúng cho ảnh 7. Loss chia trọng số cho cả hai positive. Nếu dùng mỗi đường chéo làm nhãn, mô hình sẽ bị phạt vì tìm được mô tả đúng thứ hai.

## Chạy offline

```bash
python -m vision_ml.retrieve --out runs/retrieval
python -m vision_ml.anomaly --out runs/anomaly
```

Để train retrieval với feature thật, chuẩn bị hai NPZ khác image IDs. Mỗi file có `image[N,Di]`, `text[N,Dt]` kiểu float và `image_id[N]` kiểu Unicode hoặc số, một hàng cho mỗi caption. Với cùng image ID, lặp lại cùng vector ảnh. Chạy `python -m vision_ml.retrieve --train data/train.npz --valid data/valid.npz`. Chi phí đánh giá là ma trận số caption × số ảnh; bài thi nên bắt đầu với gallery cố định vừa RAM.

Với anomaly, dùng `python -m vision_ml.anomaly --category data/mvtec/bottle`. Loader đọc `train/good` và `test/good` cùng các thư mục lỗi trong `test`. Baseline CPU resize về 128², lấy mean/std RGB ở patch 8². Kết quả gồm scores.csv và heatmaps.npy. Thay descriptor bằng feature CNN/DINO khi đã kiểm tra tọa độ heatmap. Chỉ dùng test labels khi đánh giá cuối.

## Tái lập bài Omniglot

Giải nén `images_evaluation.zip` của tác giả, rồi chạy `python -m vision_ml.oneshot data/images_evaluation --way 20 --shot 1 --episodes 200`. Dataset có dạng `alphabet/character/*.png`. [Dữ liệu chính thức](https://github.com/brendenlake/omniglot/tree/master/python)

Baseline thu ảnh về 28², đảo nền, lấy cosine với prototype. Trên **659 ký tự evaluation**, 200 episode 20-way 1-shot với seed 7 đạt **29,20% accuracy**, standard error theo episode **0,76 điểm phần trăm**. Chương trình lấy các lớp trên toàn bộ evaluation alphabets; các công bố dùng danh sách episode cố định hoặc distractor cùng alphabet cần đối chiếu riêng.

## 30 challenge theo thứ tự ưu tiên ôn

| Ưu tiên | Challenge | Metric cần theo dõi |
|---:|---|---|
| 1 | [MVTec LOCO AD · Phát hiện sai logic](challenges/01-loco.md) | Image AUROC ↑, tách structural/logical |
| 2 | [MVTec AD · Defect chưa từng thấy](challenges/02-mvtec.md) | Image AUROC, pixel AUROC và AUPRO ↑ |
| 3 | [VisA · Nhiều đối tượng trong ảnh kiểm lỗi](challenges/03-visa.md) | Image AUROC và pixel average precision ↑ cho bài luyện |
| 4 | [miniImageNet · Phân lớp 5-way 1-shot](challenges/04-mini.md) | Episode accuracy ↑, báo mean và khoảng tin cậy qua episode |
| 5 | [Omniglot · Nhận dạng ký tự mới](challenges/05-omniglot.md) | 20-way 1-shot episode accuracy ↑ |
| 6 | [CUB · Generalized zero-shot bằng thuộc tính](challenges/06-cubzsl.md) | Harmonic mean của per-class accuracy seen/unseen ↑ |
| 7 | [Stanford Online Products · Tìm đúng sản phẩm](challenges/07-sop.md) | Recall@1 và Recall@10 ↑ |
| 8 | [Revisited Oxford · Truy hồi landmark](challenges/08-roxford.md) | mAP ↑ riêng Medium và Hard |
| 9 | [DomainNet · Chuyển miền ảnh vẽ](challenges/09-domainnet.md) | Top-1 accuracy miền đích ↑ |
| 10 | [Waterbirds · Tách loài chim khỏi nền](challenges/10-waterbirds.md) | Worst-group accuracy ↑ trên 4 nhóm |
| 11 | [iWildCam · Phân loại ở camera mới](challenges/11-iwild.md) | Macro-F1 ↑ theo WILDS |
| 12 | [FMoW-WILDS · Phân loại công trình khi đổi vùng](challenges/12-fmow.md) | Worst-region accuracy ↑ cùng average accuracy |
| 13 | [PovertyMap-WILDS · Hồi quy từ ảnh vệ tinh](challenges/13-poverty.md) | Pearson r ↑, báo r cho urban/rural và nhóm thấp nhất |
| 14 | [BigEarthNet · Gán nhiều nhãn lớp phủ](challenges/14-sat.md) | Micro và macro average precision ↑ cho bài luyện |
| 15 | [LEVIR-CD · Tìm tòa nhà thay đổi](challenges/15-levir.md) | Foreground F1 và IoU ↑ |
| 16 | [AID · Scene classification khi ít nhãn](challenges/16-aid.md) | Overall accuracy và macro-F1 ↑ |
| 17 | [Flickr30k · Tìm ảnh theo mô tả](challenges/17-flickr.md) | R@1/5/10 ↑ báo tách hai chiều |
| 18 | [Flickr30k Entities · Ground phrase vào box](challenges/18-entities.md) | Recall@1/5 ↑ với IoU ≥0.5 |
| 19 | [CLEVR · Suy luận quan hệ và đếm](challenges/19-clevr.md) | Accuracy ↑ theo count, compare, exist, query |
| 20 | [NLVR2 · Kiểm tra mệnh đề trên hai ảnh](challenges/20-nlvr.md) | Accuracy và consistency theo statement ↑ |
| 21 | [TextVQA · Đọc chữ để trả lời ảnh](challenges/21-textvqa.md) | VQA soft accuracy ↑ theo evaluator |
| 22 | [DocVQA · Trả lời trên tài liệu scan](challenges/22-doc.md) | ANLS ↑ theo scorer DocVQA |
| 23 | [ChartQA · Đọc biểu đồ và tính toán](challenges/23-chart.md) | Relaxed accuracy ↑ theo evaluator ChartQA |
| 24 | [FUNSD · Liên kết key và value](challenges/24-funsd.md) | Micro-F1 trên cạnh ↑ ở bài luyện với entity vàng |
| 25 | [VCR · Chọn đáp án có căn cứ trực quan](challenges/25-vcr.md) | Q→A accuracy ↑ |
| 26 | [Charades-STA · Tìm khoảng thời gian theo câu](challenges/26-charades.md) | R@1 tại tIoU 0.5 và 0.7 ↑ |
| 27 | [MSR-VTT · Tìm video từ mô tả](challenges/27-msrvtt.md) | Text-to-video Recall@1/5/10 ↑ |
| 28 | [AVE · Định vị sự kiện nghe và thấy](challenges/28-ave.md) | Segment accuracy ↑ ở supervised setting |
| 29 | [VGGSound · Nhận dạng sự kiện audio-visual](challenges/29-vggsound.md) | Top-1 và top-5 accuracy ↑ |
| 30 | [GQA · Hỏi đáp dựa trên scene graph](challenges/30-gqa.md) | Accuracy trên balanced split ↑ |
