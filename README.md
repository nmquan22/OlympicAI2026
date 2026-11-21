# Olympic AI 2026 · Sổ tay thực hành 5 tiếng

**90 bài luyện, 3 mảng, code Python và giải thích tiếng Việt.** Mỗi bài có nguồn dữ liệu, đầu vào/đầu ra, metric, cách rút gọn trong 5 tiếng, hình cắt từ paper, sơ đồ kiến trúc và một phép thử giúp quyết định bước nâng cấp tiếp theo.

| Mảng | Bắt đầu bằng | Khi đã có baseline |
|---|---|---|
| [NLP · 30 bài](nlp/README.md) | UIT-VSFC, PhoATIS, CoNLL-2003 | QA tiếng Việt, retrieval, fact verification, decoder-only và text-to-SQL |
| [Computer Vision · 30 bài](vision/README.md) | Oxford-IIIT Pet, Pascal VOC, VisDrone | Thermal/UAV có thể dùng lại các khối bbox và multi-scale; tiếp tục với tracking, restoration và point cloud |
| [CV + ML và đa phương thức · 30 bài](vision_ml/README.md) | MVTec LOCO AD, MVTec AD, VisA | Few-shot, domain shift, image–text retrieval, grounding và audio–video |

Thứ tự trong từng mảng là ưu tiên ôn tập do người biên soạn chọn. Bài đứng đầu có baseline dễ kiểm tra, có lỗi điển hình nhìn thấy được và có một cải tiến vừa sức thử trong buổi thi. Một số mục là shared task có xếp hạng chính thức; số khác là benchmark học thuật được chuyển thành đề luyện. Mỗi trang ghi điều kiện của mốc nghiên cứu. Thứ hạng và kết quả của paper gắn với thời điểm tác giả công bố.

## Ba bài nên mở đầu tiên

- [VisDrone-DET](vision/challenges/03-visdrone.md) — Xem bbox của vật thể nhỏ sau resize, rồi so inference nguyên khung với cắt tile. [Code đọc và augment bbox](vision/blocks/data.py)
- [MVTec LOCO AD](vision_ml/challenges/01-loco.md) — So local patch score với histogram toàn ảnh để phát hiện trường hợp mọi chi tiết trông bình thường nhưng sản phẩm thiếu một món. [Code patch memory](vision_ml/blocks/anomaly.py)
- [PhoATIS](nlp/challenges/02-phoatis.md) — Một câu có thể đúng intent nhưng sai cả khung vì mất một slot. Bài tập chỉ ra cách nối hai head và cách chấm span. [Code encoder và decoder-only](nlp/blocks/transformers.py)

## Chạy từ thư mục gốc

Đã kiểm tra với Python 3.12, PyTorch 2.5.1 CPU và torchvision 0.20.1 CPU. Khi dùng GPU, cài cặp torch/torchvision tương thích với CUDA của máy. Nếu thi offline, chuẩn bị wheel và dataset trước buổi thi.

```bash
python -m pip install -r requirements.txt
python -m nlp.train --mode encoder --out runs/encoder
python -m nlp.train --mode lm --out runs/lm
python -m vision.detect --out runs/detection
python -m vision.segment --out runs/segment
python -m vision_ml.retrieve --out runs/retrieval
python -m vision_ml.anomaly --out runs/anomaly
```

Các lệnh trên tự tạo hoặc dùng fixture nhỏ và chạy được khi đã ngắt mạng. Tham số cho dữ liệu thật và định dạng input nằm ở README của từng mảng. Với Parquet hoặc evaluator COCO, cài thêm `python -m pip install -r requirements-extras.txt`. Mỗi `blocks/` chứa các thành phần tái sử dụng; mỗi `challenges/` chứa 30 đề và hướng giải.

## Kết quả đã chạy trên CPU bằng dữ liệu thật

| Bài | Dữ liệu đánh giá | Kết quả | Cách tái lập |
|---|---|---|---|
| UIT-VSFC | 6.000 mẫu train, 1.583 mẫu validation chính thức | TF-IDF ký tự + từ đạt **0,7448 macro-F1**, accuracy 0,8901 | [Lệnh và data](nlp/README.md) |
| RetinaMNIST 28² | Train và validation chính thức | Ordinal logistic đạt **0,5333 accuracy**, MAE **0,8417**. Softmax đạt 0,5000 và 0,8833 | [Lệnh và data](vision/README.md) |
| Omniglot | 659 ký tự evaluation, 200 episode tự lấy với seed 7, 20-way 1-shot | Pixel prototype đạt **29,20% episode accuracy** | [Lệnh và data](vision_ml/README.md) |

Các bảng trong paper là kết quả của tác giả. Các hướng nâng cấp còn lại là bài để tự thử. Phép kiểm tra code còn bao gồm causal mask, padding, VOC/YOLO/COCO tọa độ, flip và resize với bbox rỗng, mask segmentation và retrieval có nhiều positive.

## Chia 300 phút cho một đề

| Phút | Việc cần hoàn thành | Đầu ra |
|---|---|---|
| 0–30 | Đọc evaluator, schema và vài mẫu; chia đúng đơn vị ảnh, video, document hoặc ID | Manifest train/dev và một batch đã kiểm tra |
| 30–90 | Chạy baseline rẻ nhất trên subset; xuất thử prediction theo format đề | Một điểm dev và một file prediction đọc lại được |
| 90–150 | Tách lỗi theo nhóm ghi ở trang challenge | Danh sách ví dụ lỗi và một giả thuyết |
| 150–235 | Thay đúng một bước, đo cùng split/seed và số mẫu | Bảng baseline–ablation, thời gian và bộ nhớ |
| 235–275 | Chạy cấu hình tốt nhất trong ngân sách còn lại | Checkpoint, cấu hình inference và predictions |
| 275–300 | Kiểm tra số dòng, ID, thứ tự, hệ tọa độ hoặc tokenizer. Chạy lại lệnh xuất | File nộp cuối và một lệnh tái lập |

Đây là ngân sách luyện tập mỗi bài. Thời gian thực tế phụ thuộc kích thước subset, checkpoint đã cache và GPU được cấp. Với leaderboard có điều kiện khác bài rút gọn, dùng nó để chọn thí nghiệm và so các model của mình trên cùng split.
