# Workshop và shared task 2026 để luyện Olympic AI

Đối chiếu nguồn ngày **2026-11-22**: **39 hướng luyện từ 27 sự kiện**.

Đây là danh mục chọn lọc theo độ hữu ích cho bài thi 5 tiếng. A = nên luyện trước, B = mở rộng độ phủ, C = khám phá khi bài cấp đủ dữ liệu/weights. Nhãn này là đánh giá của người biên soạn, không phải xác suất ra đề hay khuyến nghị chính thức của IOAI.

Cách chọn ưu tiên: đối chiếu kỹ năng với đề IOAI đã công bố/syllabus đúng năm; khả năng tạo baseline bằng model cơ bản hoặc feature được cấp; tốc độ vòng lặp train–dev; và mức bổ sung cho các mảng ít được luyện. Bài cần simulation, spatial audio hoặc generative weights lớn thường ở C. Chưa có dữ liệu thống kê về tần suất đề Olympic AI Việt Nam để gán phần trăm.

Nhiều dòng là các track của cùng workshop. Một số sự kiện là challenge/conference, đã ghi rõ trên từng thẻ. **39 hướng không có nghĩa là 39 workshop mới thành lập**. Chỉ gắn nhãn lần đầu khi nguồn xác nhận.

Mỗi thẻ tách sự kiện đã xác minh, đề luyện tự biên soạn, baseline, nâng cấp và hướng đọc thêm. Metric bài luyện không thay scorer challenge. Code liên kết là khối nền hoặc baseline rút gọn; chưa tái lập giải pháp thắng/SOTA 2026.

Xem thêm [đề IOAI 2026](IOAI_2026.md), [bản đồ khối code](BLOCKS.md), [kế hoạch luyện](PRACTICE_PLAN.md).

## ml

| Mức | Hướng luyện | Sự kiện |
|---|---|---|
| A | [TSALM: forecasting có context](workshops/tsalm.md) | TSALM / ICLR 2026 |
| B | [TS4H: chuỗi không đều, missingness và subject split](workshops/ts4h.md) | TS4H / ICLR 2026 |

## nlp

| Mức | Hướng luyện | Sự kiện |
|---|---|---|
| A | [SemEval 3: sentiment theo aspect và cường độ](workshops/semeval3.md) | SemEval / ACL 2026 |
| A | [SemEval 4: tương đồng cốt truyện](workshops/semeval4.md) | SemEval / ACL 2026 |
| A | [SemEval 8: retrieval hội thoại nhiều lượt](workshops/semeval8.md) | SemEval / ACL 2026 |
| A | [SemEval 11: logic tách khỏi content bias](workshops/semeval11.md) | SemEval / ACL 2026 |
| A | [CODI-CRAC: discourse, coreference và thứ tự](workshops/codi-crac.md) | CODI-CRAC / ACL 2026 |
| A | [SURGeLLM: bảng, schema và truy hồi có cấu trúc](workshops/surgellm.md) | SURGeLLM / ACL 2026 |
| B | [LAW XX: nhãn nhiễu và annotation](workshops/law20.md) | LAW / ACL 2026 |
| B | [BEA: chấm điểm bài viết và feedback](workshops/bea26.md) | BEA / ACL 2026 |
| B | [TrustNLP: robustness và confidence](workshops/trustnlp26.md) | TrustNLP / ACL 2026 |
| B | [RAG4Reports: báo cáo có dẫn evidence](workshops/rag4reports26.md) | RAG4Reports / ACL 2026 |
| B | [WMT: phát hiện lỗi dịch và span](workshops/wmt26-quality.md) | WMT 2026 |

## vision

| Mức | Hướng luyện | Sự kiện |
|---|---|---|
| A | [NTIRE: restoration và đánh giá chất lượng](workshops/ntire-restoration.md) | NTIRE / CVPR 2026 |
| A | [PBVS: thermal SR có RGB hướng dẫn](workshops/pbvs-thermal-sr.md) | PBVS / CVPR 2026 |
| B | [PBVS: SR ảnh hyperspectral thực](workshops/pbvs-hyperspectral.md) | PBVS / CVPR 2026 |
| A | [UG2+: segmentation trong thời tiết xấu](workshops/ug2-weather.md) | UG2+ / CVPR 2026 |
| C | [GenRecon3D: hình học và độ bất định](workshops/genrecon3d.md) | GenRecon3D / CVPR 2026 |
| C | [LoViF: restoration với reward và mô hình sinh](workshops/lovif.md) | LoViF / CVPR 2026 |

## vision_ml

| Mức | Hướng luyện | Sự kiện |
|---|---|---|
| B | [PBVS: phân đoạn sạt lở trên Sao Hỏa](workshops/pbvs-mars.md) | PBVS / CVPR 2026 |
| A | [PBVS: EO–SAR, thiếu modality và OOD](workshops/pbvs-eo-sar.md) | PBVS / CVPR 2026 |
| A | [VAND 4.0: anomaly dưới thay đổi ánh sáng và pose](workshops/vand4.md) | VAND 4.0 / CVPR 2026 |
| B | [VISION: kiểm tra công nghiệp và ảnh X-ray](workshops/vision26.md) | VISION / CVPR 2026 |
| A | [ECV/LPCVC: retrieval và video với ngân sách inference](workshops/ecv-lpcvc.md) | ECV / CVPR 2026 |
| A | [DataCV: chọn dữ liệu và chất lượng nhãn](workshops/datacv.md) | DataCV / CVPR 2026 |
| A | [GRAIL-V: retrieval có evidence và reranking](workshops/grail-v.md) | GRAIL-V / CVPR 2026 |
| A | [EgoVis: tìm hành động trong video bằng câu mô tả](workshops/egovis.md) | EgoVis / CVPR 2026 |
| B | [EarthVision: ảnh vệ tinh, split theo vùng](workshops/earthvision.md) | EarthVision / CVPR 2026 |
| B | [CV4Smalls: video với ít nhãn](workshops/cv4smalls.md) | CV4Smalls / CVPR 2026 |
| B | [SELVA: distillation và hiệu quả đa phương thức](workshops/selva.md) | SELVA / ACL 2026 |
| B | [ALVR: hiểu quan hệ ngôn ngữ–ảnh](workshops/alvr26.md) | ALVR / ACL 2026 |
| A | [MAGMaR: truy hồi đa phương thức trước generation](workshops/magmar26.md) | MAGMaR / ACL 2026 |

## audio

| Mức | Hướng luyện | Sự kiện |
|---|---|---|
| A | [DCASE 1: phân loại âm thanh theo taxonomy](workshops/dcase1.md) | DCASE 2026 |
| A | [DCASE 2: anomalous sound, nhiễu và domain shift](workshops/dcase2.md) | DCASE 2026 |
| C | [DCASE 3: semantic acoustic imaging](workshops/dcase3.md) | DCASE 2026 |
| C | [DCASE 4: tách và phân đoạn sound scene](workshops/dcase4.md) | DCASE 2026 |
| B | [DCASE 5: audio QA phải nghe mới trả lời được](workshops/dcase5.md) | DCASE 2026 |
| A | [DCASE 6: tìm đoạn trong audio dài](workshops/dcase6.md) | DCASE 2026 |
| B | [DCASE 7: continual learning không replay dữ liệu cũ](workshops/dcase7.md) | DCASE 2026 |

## Dùng danh mục offline

```bash
python -m catalog.search --area ml --priority A
python -m catalog.search --query retrieval
python tools/render_catalog.py
```

Dữ liệu gốc nằm ở [workshops_2026.json](../catalog/workshops_2026.json). Sửa registry rồi render lại để bảng và thẻ không lệch nhau. Các URL không được gọi tự động trong lúc chạy baseline.
