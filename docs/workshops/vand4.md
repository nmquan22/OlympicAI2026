# VAND 4.0: anomaly dưới thay đổi ánh sáng và pose

**VAND 4.0 / CVPR 2026 · workshop · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Hai track 2026 dùng MVTec AD 2 và Kaputt 2. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://sites.google.com/view/vand4-cvpr2026/challenge).

## Đề luyện do repo đề xuất

Ảnh kiểm tra + normal reference → score/heatmap lỗi.

Metric để luyện: **AUROC/AP và lỗi theo điều kiện cho bài luyện; dùng scorer track khi so challenge.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Patch kNN và score toàn ảnh trên feature được phép. |
| Nâng cấp vừa sức | Memory đa tỉ lệ, chuẩn hóa theo condition; ablate local với global context. |
| Hướng nghiên cứu để đọc tiếp | Đọc training-free localization và adaptation dưới lighting shift. |

Phép thử quyết định bước tiếp theo: **Cố định memory budget; báo lỗi theo lighting/pose thay vì chỉ average.**

## Chuẩn bị và khối code

Khác VAND 3.0/LOCO 2025. Chưa tải dataset hoặc tái lập các giải pháp thắng 2026.

- [vision_ml/blocks/anomaly.py](../../vision_ml/blocks/anomaly.py)
- [ml/blocks/validation.py](../../ml/blocks/validation.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
