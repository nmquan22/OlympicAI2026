# Olympic AI 2026 · Sổ tay thực hành 5 tiếng

**Ưu tiên hiện tại: [5 bài NLP AICC để học idea](experiments/nlp_aicc/README.md)** — đọc reference trước, 7 thí nghiệm T4/bài, BPE và model scratch theo điều kiện đề, tối đa 2 version cùng chạy. Có phân tích baseline → reference → giả thuyết mới; kế hoạch chưa phải kết quả đã chạy.

**AICC đã chạy thử trên Kaggle:** [Cosmos Cosmic Run](experiments/aicc/cosmos-cosmic-run/REPORT.md) — public **0.72343**, private **0.71387**, luyện tập sau cuộc thi. Có notebook, log, ablation và [hàng đợi 10 bài](experiments/aicc/queue.json), giới hạn 2 phiên đồng thời; mới hoàn tất smoke test 1 bài.

**Mới 24/11: [16 bài Vision nâng cao](vision/curriculum/README.md)** — notebook có giải thích tiếng Việt theo từng cell. Bắt đầu bằng [Slot Attention, 11 cell](vision/curriculum/CV01_CELL_BY_CELL_VI.md); tiếp tục với [CV02–CV16](vision/curriculum/CELL_BY_CELL_VI.md). Dùng torchvision/PyTorch cho các khối có sẵn; fixture chỉ kiểm tra cơ chế, chưa phải benchmark/SOTA.

**5 mảng, 90 bài nền tảng, phân tích 6 đề IOAI 2026 và 39 hướng từ 27 workshop/sự kiện 2026.** Code Python tiếng Anh, giải thích tiếng Việt. Bổ sung ngày **22/11/2026**: ML bảng, time series, audio, suy luận tương tác, robustness và các khối đa phương thức.

Mở [danh mục workshop 2026](docs/WORKSHOPS_2026.md), [đề IOAI và syllabus mới](docs/IOAI_2026.md), [bản đồ khối code](docs/BLOCKS.md), [18 buổi luyện 5 tiếng](docs/PRACTICE_PLAN.md) và [phạm vi kiểm thử](docs/VALIDATION.md). Mỗi hướng mới có baseline, nâng cấp, phép thử quyết định và link code; các bài nền tảng giữ hình paper và sơ đồ sẵn có.

| Mảng | Bắt đầu bằng | Khi đã có baseline |
|---|---|---|
| [NLP · 30 bài](nlp/README.md) | UIT-VSFC, PhoATIS, CoNLL-2003 | QA tiếng Việt, retrieval, fact verification, decoder-only và text-to-SQL |
| [Computer Vision · 30 bài](vision/README.md) | Oxford-IIIT Pet, Pascal VOC, VisDrone | Thermal/UAV có thể dùng lại các khối bbox và multi-scale; tiếp tục với tracking, restoration và point cloud |
| [CV + ML và đa phương thức · 30 bài](vision_ml/README.md) | MVTec LOCO AD, MVTec AD, VisA | Few-shot, domain shift, image–text retrieval, grounding và audio–video |
| [ML bảng, chuỗi thời gian và suy luận](ml/README.md) | Linear/tree pipeline, seasonal naive, ridge lag | Graph, behavior cloning, semantic comparison search; RL mở rộng 2027 |
| [Audio](audio/README.md) | WAV/log-mel, sound CNN, CTC decoding | Audio anomaly, audio QA, temporal grounding, continual learning |

Ưu tiên A/B/C trong danh mục là **đánh giá ôn tập**, không phải xác suất ra đề. Nhiều hướng thuộc cùng một workshop; chỉ ghi “lần đầu” khi nguồn xác nhận. Syllabus ở URL `2026/11` thực tế mang tên **IOAI 2027**. VAND 4.0 năm 2026 dùng **MVTec AD 2/Kaputt 2**, phân biệt với bài LOCO/VAND 3.0 năm 2025.

## Các hướng nên thêm vào lịch ôn

| Chưa quen dạng nào | Mở bài nào | Khối bắt đầu |
|---|---|---|
| ML có cấu trúc và leakage | [TSALM](docs/workshops/tsalm.md), [TS4H](docs/workshops/ts4h.md) | [tabular](ml/tabular.py), [forecast](ml/forecast.py) |
| Tối ưu thứ tự/ranh giới | [Find the Order, Ghost](docs/IOAI_2026.md) | [ordering](nlp/blocks/ordering.py), [changepoint](nlp/blocks/changepoint.py) |
| Suy luận chống shortcut | [SemEval 11](docs/workshops/semeval11.md), [DCASE 5](docs/workshops/dcase5.md) | Entity-swap, silence/shuffle control |
| Audio không cần ASR | [DCASE 1/2/6](audio/README.md) | [log-mel + CNN](audio/classify.py), [interval decoder](audio/blocks/temporal.py) |
| Thiếu modality và domain shift | [PBVS EO–SAR](docs/workshops/pbvs-eo-sar.md), [UG2+](docs/workshops/ug2-weather.md) | [masked fusion](vision_ml/blocks/multimodal.py), group split |
| Tương tác và tối ưu theo metric | [Potato, Double Agent, IOAI Field](docs/IOAI_2026.md) | Query selection, selective gradients, coordinate field |

Thứ tự trong từng mảng là ưu tiên ôn tập do người biên soạn chọn. Bài đứng đầu có baseline dễ kiểm tra, có lỗi điển hình nhìn thấy được và có một cải tiến vừa sức thử trong buổi thi. Một số mục là shared task có xếp hạng chính thức; số khác là benchmark học thuật được chuyển thành đề luyện. Mỗi trang ghi điều kiện của mốc nghiên cứu. Thứ hạng và kết quả của paper gắn với thời điểm tác giả công bố.

## Ba bài nền tảng gốc

- [VisDrone-DET](vision/challenges/03-visdrone.md) — Xem bbox của vật thể nhỏ sau resize, rồi so inference nguyên khung với cắt tile. [Code đọc và augment bbox](vision/blocks/data.py)
- [MVTec LOCO AD](vision_ml/challenges/01-loco.md) — So local patch score với histogram toàn ảnh để phát hiện trường hợp mọi chi tiết trông bình thường nhưng sản phẩm thiếu một món. [Code patch memory](vision_ml/blocks/anomaly.py)
- [PhoATIS](nlp/challenges/02-phoatis.md) — Một câu có thể đúng intent nhưng sai cả khung vì mất một slot. Bài tập chỉ ra cách nối hai head và cách chấm span. [Code encoder và decoder-only](nlp/blocks/transformers.py)

## Chạy từ thư mục gốc

Các baseline gốc từng được kiểm tra với Python 3.12, PyTorch 2.5.1 CPU và torchvision 0.20.1 CPU. Phiên bản kiểm tra đợt mở rộng nằm trong [VALIDATION.md](docs/VALIDATION.md). Khi dùng GPU, cài cặp torch/torchvision tương thích với CUDA của máy. Nếu thi offline, chuẩn bị wheel và dataset trước buổi thi.

```bash
python -m pip install -r requirements.txt
python -m nlp.train --mode encoder --out runs/encoder
python -m nlp.train --mode lm --out runs/lm
python -m vision.detect --out runs/detection
python -m vision.segment --out runs/segment
python -m vision_ml.retrieve --out runs/retrieval
python -m vision_ml.anomaly --out runs/anomaly
python -m ml.tabular --out runs/tabular
python -m ml.forecast --out runs/forecast
python -m audio.classify --out runs/audio
python -m nlp.boundary --out runs/boundary
python -m nlp.order --out runs/order/answers.json
python -m vision.restore --out runs/restoration
python -m ml.q_learning
python -m catalog.search --priority A
```

Các lệnh trên tự tạo hoặc dùng fixture nhỏ và chạy được khi đã ngắt mạng. **Điểm fixture chỉ kiểm tra đường chạy, không phải kết quả benchmark.** Tham số cho dữ liệu thật và định dạng input nằm ở README của từng mảng. Với Parquet hoặc evaluator COCO, cài thêm `python -m pip install -r requirements-extras.txt`. Ba mảng gốc có `challenges/` với 30 bài/mảng; hướng mới tập trung ở `docs/workshops/`, tra được offline qua registry JSON.

```bash
python tools/check_repo.py
python -m unittest discover -s tests -v
python tools/smoke_recipes.py
```

Khối mới được ghi rõ là chương trình trọn luồng, thành phần cần adapter hay hướng đọc thêm trong [BLOCKS.md](docs/BLOCKS.md). Chưa tái lập leaderboard/SOTA của 39 hướng workshop. Baseline mới không tự tải weights hay dataset; cần chuẩn bị wheel/data/weights hợp lệ trước khi thi offline.

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
