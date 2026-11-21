# EarthVision: ảnh vệ tinh, split theo vùng

**EarthVision / CVPR 2026 · workshop · ưu tiên B · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

EarthVision được liệt kê trong chương trình workshop CVPR 2026. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://cvpr.thecvf.com/Conferences/2026/Workshops).

## Đề luyện do repo đề xuất

Ảnh vệ tinh theo vùng/thời gian → lớp, mask hoặc thay đổi.

Metric để luyện: **Macro-F1/mIoU theo vùng trên bài luyện.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Linear head từ encoder được phép; hoặc U-Net nhỏ. |
| Nâng cấp vừa sức | Metadata thời gian, feature chênh lệch trước/sau, balanced sampler. |
| Hướng nghiên cứu để đọc tiếp | Đọc cross-region adaptation; chọn đúng nhiệm vụ từ dữ liệu đề. |

Phép thử quyết định bước tiếp theo: **Giữ nguyên mô hình; so random split và spatial holdout.**

## Chuẩn bị và khối code

Chỉ xác minh workshop qua danh mục chính thức; chưa gán dataset/track 2026 cụ thể.

- [ml/blocks/validation.py](../../ml/blocks/validation.py)
- [vision_ml/blocks/tabular.py](../../vision_ml/blocks/tabular.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
