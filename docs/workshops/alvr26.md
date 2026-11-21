# ALVR: hiểu quan hệ ngôn ngữ–ảnh

**ALVR / ACL 2026 · workshop · ưu tiên B · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

ALVR 2026 là workshop language–vision, gồm grounded understanding. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://2026.aclweb.org/program/workshops/).

## Đề luyện do repo đề xuất

Region features + câu hỏi → đáp án hoặc vùng evidence.

Metric để luyện: **Accuracy theo loại câu hỏi; grounding Recall@K cho bài luyện.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Global image pooling + text feature. |
| Nâng cấp vừa sức | SpatialQuestionHead và hard negative quan hệ. |
| Hướng nghiên cứu để đọc tiếp | Đọc grounded compositional reasoning. |

Phép thử quyết định bước tiếp theo: **Hoán đổi trái/phải hoặc số lượng trong câu; kiểm tra model có phân biệt.**

## Chuẩn bị và khối code

Workshop đã xác minh qua ACL; dùng CLEVR/GQA trong repo làm dữ liệu luyện, không gọi chúng là dataset mới 2026.

- [vision_ml/blocks/fusion.py](../../vision_ml/blocks/fusion.py)
- [vision_ml/blocks/multimodal.py](../../vision_ml/blocks/multimodal.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
