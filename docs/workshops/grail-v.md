# GRAIL-V: retrieval có evidence và reranking

**GRAIL-V / CVPR 2026 · workshop · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Workshop tập trung retrieval, reranking và verification đa phương thức. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://grailworkshops.github.io/).

## Đề luyện do repo đề xuất

Câu hỏi + ảnh/tài liệu/clip → evidence được xếp hạng.

Metric để luyện: **nDCG/R@K và evidence accuracy của bài luyện.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Sparse hoặc cosine retrieval trên corpus được cấp. |
| Nâng cấp vừa sức | RRF sparse+dense rồi rerank top-k; lưu ID evidence. |
| Hướng nghiên cứu để đọc tiếp | Đọc verification có calibration; chỉ thêm generator nếu evaluator cần. |

Phép thử quyết định bước tiếp theo: **Đo oracle evidence với retrieved evidence để biết bottleneck nằm ở retrieval hay answer.**

## Chuẩn bị và khối code

Dùng subset train/dev và các weights được đề thi cho phép; chuẩn bị dữ liệu trước buổi luyện.

- [nlp/blocks/retrieval.py](../../nlp/blocks/retrieval.py)
- [vision_ml/blocks/fusion.py](../../vision_ml/blocks/fusion.py)
- [common/ensemble.py](../../common/ensemble.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
