# WMT: phát hiện lỗi dịch và span

**WMT 2026 · shared_task · ưu tiên B · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

WMT 2026 có đánh giá chất lượng dịch, gồm lỗi ở mức span. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://www2.statmt.org/wmt26/mteval-task.html).

## Đề luyện do repo đề xuất

Source + hypothesis → score hoặc vị trí/severity lỗi.

Metric để luyện: **Span-F1, MAE hoặc correlation trong bài luyện tùy output.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Encoder pair head hoặc token head với offsets gốc. |
| Nâng cấp vừa sức | Thêm features length ratio và alignment; decode span có ràng buộc. |
| Hướng nghiên cứu để đọc tiếp | Đọc quality estimation không cần reference và multilingual transfer. |

Phép thử quyết định bước tiếp theo: **Chia dev theo system/domain; tránh ghi nhớ artifact của hệ thống dịch.**

## Chuẩn bị và khối code

WMT là conference/shared-task series; không gộp thành một workshop CVPR.

- [nlp/blocks/structured.py](../../nlp/blocks/structured.py)
- [nlp/blocks/transformers.py](../../nlp/blocks/transformers.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
