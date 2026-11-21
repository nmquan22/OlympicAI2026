# Computer Vision · 30 bài luyện

Ba bài đầu cho ba kiểu nhãn thường gặp. Oxford-IIIT Pet dùng mask có lớp border. VOC lưu hộp theo hai góc. VisDrone lưu hộp theo góc trên trái và kích thước, kèm thông tin vùng bỏ qua. Đọc hình dự đoán trong hệ tọa độ gốc trước khi so model.

## Học code theo thứ tự

| Khối | Đầu vào → đầu ra | Điều cần đọc |
|---|---|---|
| [COCO, VOC và YOLO](blocks/data.py) | annotation → float XYXY và class IDs | Readers, mapping category, ảnh âm và collate danh sách box biến độ dài |
| [Augmentation ảnh và bbox](blocks/data.py) | image + BoundingBoxes → cặp đã resize/flip | Hình học áp dụng chung, color jitter chỉ đổi pixels, sanitize bỏ hộp suy biến và nhãn tương ứng |
| [Dataset cho mask](blocks/masks.py) | CSV image/mask → tensor ảnh và class map | Nearest-neighbor giữ nguyên class ID và ignore 255 |
| [U-Net và Faster R-CNN nhỏ](blocks/models.py) | ảnh → logits mask hoặc proposals → ROI predictions | Skip connection, anchor sizes, ROI pooling và số class kể cả nền |
| [Train detection và xuất COCO](detect.py) | COCO JSON → detections.json theo tọa độ gốc | RPN/ROI losses, resize ngược và hoàn nguyên category ID |
| [Chấm COCO](eval_coco.py) · [Train segmentation](segment.py) | prediction + nhãn → AP hoặc mIoU | Dùng bộ chấm COCO chính thức cho COCO. Semantic segmentation tích lũy confusion matrix toàn dev |
| [Ghép track bằng IoU](blocks/tracking.py) | hai tập hộp → cặp chỉ số | Hungarian với ngưỡng IoU, hỗ trợ frame rỗng |
| [Ordinal regression](ordinal.py) | ảnh RetinaMNIST → xác suất 5 bậc | So softmax với bốn xác suất vượt ngưỡng |

## Một bbox qua ba định dạng

Với ảnh **100 × 50** và hộp phủ toàn khung, cả ba reader đưa về cùng một hộp `[0, 0, 100, 50]`. Trong hệ tọa độ cạnh ảnh, cạnh phải có thể bằng chiều rộng.

| File nguồn | Bốn số đầu vào | Phép chuyển |
|---|---|---|
| COCO XYWH theo pixel | `[0, 0, 100, 50]` | Cộng w, h vào góc trái |
| Pascal VOC XYXY, 1-based inclusive | `[1, 1, 100, 50]` | Trừ 1 ở xmin, ymin |
| YOLO CXCYWH normalized | `[0.5, 0.5, 1, 1]` | Nhân kích thước ảnh, rồi lấy tâm ± nửa cạnh |

Với box bất đối xứng `[10, 5, 30, 20]`, horizontal flip trên ảnh rộng 100 cho `[70, 5, 90, 20]`. Resize từ 100 × 50 thành 200 × 100 cho `[20, 10, 60, 40]` trước khi flip. Thử vẽ hai hộp này trước khi bật augmentation ngẫu nhiên.

## Chạy offline

```bash
python -m vision.detect --out runs/detection
python -m vision.eval_coco runs/detection/fixture/valid.json runs/detection/detections.json
python -m vision.segment --out runs/segment
```

Detector học từ đầu. Lệnh đầu sinh ảnh có hộp và cả ảnh nền. Chạy trên dữ liệu thật bằng `python -m vision.detect --images data/images --train data/train.json --valid data/valid.json --epochs 5 --device cuda`. COCO JSON cần `images`, `annotations`, `categories`; ghi width/height gốc đúng theo file ảnh. Bản train bỏ crowd annotations; evaluator vẫn dùng file gốc khi chấm. Dùng pycocotools từ requirements-extras cho lệnh thứ hai.

Segmentation nhận CSV gồm `image,mask`, mỗi path tương đối với CSV. Mask là PNG một kênh, class IDs từ 0 tới C−1 hoặc 255. Với trimap Oxford gốc 1,2,3, ánh xạ về 0,1,2 khi chuẩn bị file. Lệnh mẫu cho dữ liệu đã chuyển là `python -m vision.segment --train data/train.csv --valid data/valid.csv --classes 3`.

## Tái lập bài RetinaMNIST

Tải bản `retinamnist.npz` 28 × 28 của MedMNIST v2. Chạy `python -m vision.ordinal data/retinamnist.npz`. Script đọc train/validation, pool trung bình patch 4 × 4 và chuẩn hóa trên train. [File từ bản phát hành](https://zenodo.org/records/10519652/files/retinamnist.npz?download=1)

| Đầu ra | Accuracy ↑ | Macro AUC ↑ | MAE theo bậc ↓ |
|---|---:|---:|---:|
| Softmax logistic | 0,5000 | 0,7753 | 0,8833 |
| Bốn ngưỡng ordinal | 0,5333 | 0,7705 | 0,8417 |

Trên official validation, ordinal giảm khoảng cách giữa bậc dự đoán và bậc thật, còn AUC giảm nhẹ. Hai metric trả lời hai câu hỏi khác nhau.

## 30 challenge theo thứ tự ưu tiên ôn

| Ưu tiên | Challenge | Metric cần theo dõi |
|---:|---|---|
| 1 | [Oxford-IIIT Pet · Tách thú cưng khỏi nền](challenges/01-pets.md) | mIoU 3 lớp ↑ cho bài luyện |
| 2 | [VOC 2007 · Detection nhiều lớp](challenges/02-voc.md) | VOC07 11-point mAP@0.5 ↑ |
| 3 | [VisDrone-DET · Vật thể nhỏ trong ảnh UAV](challenges/03-visdrone.md) | AP@[.50:.95] ↑ theo evaluator VisDrone |
| 4 | [Cityscapes · Semantic segmentation đường phố](challenges/04-city.md) | mIoU 19 lớp ↑, bỏ void 255 |
| 5 | [CIFAR-100 · Phân loại ảnh nhỏ](challenges/05-cifar.md) | Top-1 accuracy ↑ |
| 6 | [EuroSAT · Lớp phủ bề mặt qua RGB](challenges/06-eurosat.md) | Accuracy và macro-F1 ↑ cho bài luyện |
| 7 | [DRIVE · Mạch máu mảnh](challenges/07-drive.md) | F1 và ROC-AUC |
| 8 | [DIV2K · Super-resolution ×4](challenges/08-div2k.md) | PSNR ↑ |
| 9 | [SIDD · Khử nhiễu camera thật](challenges/09-sidd.md) | PSNR và SSIM ↑, cùng không gian sRGB |
| 10 | [GoPro · Khử nhòe chuyển động](challenges/10-gopro.md) | PSNR và SSIM ↑ |
| 11 | [LOL-v1 · Tăng sáng ảnh tối](challenges/11-lol.md) | PSNR, SSIM ↑ so với ảnh chuẩn |
| 12 | [DAVIS 2016 · Theo vật thể bằng mask frame đầu](challenges/12-davis.md) | Trung bình J và F ↑ |
| 13 | [MOT17 · Ghép detection thành trajectory](challenges/13-mot.md) | IDF1, HOTA ↑ qua TrackEval |
| 14 | [OTB-100 · Single-object tracking](challenges/14-otb.md) | AUC success plot ↑ cho one-pass evaluation |
| 15 | [HPatches · Ghép điểm giữa hai góc nhìn](challenges/15-hpatch.md) | MMA@3px ↑ trên matches, corner error ↓ |
| 16 | [NYU Depth v2 · Depth từ một ảnh RGB](challenges/16-nyu.md) | AbsRel ↓ và δ1 ↑ với valid-depth mask |
| 17 | [KITTI Stereo 2015 · Disparity](challenges/17-kitti.md) | D1-all ↓, bad nếu sai >3 px và >5% trên valid mask |
| 18 | [MPI Sintel · Optical flow](challenges/18-sintel.md) | Mean endpoint error ↓, tách clean/final |
| 19 | [300-W · Landmark khuôn mặt](challenges/19-300w.md) | NME ↓ chuẩn hóa khoảng cách hai khóe mắt ngoài, ghi rõ protocol |
| 20 | [MPII · Keypoint cơ thể](challenges/20-mpii.md) | PCKh@0.5 ↑ trên khớp có nhãn |
| 21 | [UCF101 · Nhận diện hành động ngắn](challenges/21-ucf.md) | Top-1 accuracy theo video ↑ |
| 22 | [ShanghaiTech · Đếm người trong đám đông](challenges/22-shanghai.md) | MAE và RMSE ↓, tách Part A và B |
| 23 | [BSDS500 · Phát hiện đường biên](challenges/23-bsd.md) | ODS và OIS F-measure ↑ theo evaluator với tolerance |
| 24 | [CUB-200-2011 · Phân loài chim](challenges/24-cub.md) | Top-1 accuracy ↑ |
| 25 | [RetinaMNIST · Phân độ theo thang thứ tự](challenges/25-retina.md) | Accuracy và AUC theo MedMNIST |
| 26 | [ADE20K · Scene parsing nhiều lớp](challenges/26-ade.md) | mIoU ↑ tính theo các lớp được định nghĩa trước |
| 27 | [SynthText · Định vị chữ xoay](challenges/27-synth.md) | Precision, recall, H-mean ↑ với polygon IoU ≥0.5 cho bài luyện |
| 28 | [IIIT5K · Đọc chữ trong crop](challenges/28-iiit5k.md) | Word accuracy ↑, lexicon-free và cùng chính sách case |
| 29 | [ModelNet40 · Phân lớp point cloud](challenges/29-modelnet.md) | Overall accuracy và mean class accuracy ↑ |
| 30 | [ShapeNetPart · Gán nhãn bộ phận](challenges/30-shapenet.md) | Instance mIoU và class mIoU ↑ theo benchmark ShapeNetPart |
