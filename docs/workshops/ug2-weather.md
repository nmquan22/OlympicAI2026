# UG2+: segmentation trong thời tiết xấu

**UG2+ / CVPR 2026 · workshop · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Các track gồm all-weather restoration, adverse-weather segmentation và turbulence video segmentation. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://cvpr2026ug2challenge.github.io/).

## Đề luyện do repo đề xuất

Ảnh/video suy giảm → mask hoặc ảnh phục hồi theo track.

Metric để luyện: **mIoU theo weather cho bài luyện segmentation.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Train segmenter trực tiếp trên ảnh suy giảm. |
| Nâng cấp vừa sức | So augmentation suy giảm với frontend phục hồi; giữ evaluator segmentation. |
| Hướng nghiên cứu để đọc tiếp | Đọc adaptation và xử lý đồng thời phục hồi/nhận thức. |

Phép thử quyết định bước tiếp theo: **Chèn restorer trước model frozen; đo mIoU, không chọn chỉ bằng PSNR.**

## Chuẩn bị và khối code

Dùng subset train/dev và các weights được đề thi cho phép; chuẩn bị dữ liệu trước buổi luyện.

- [vision/blocks/models.py](../../vision/blocks/models.py)
- [vision/blocks/restoration.py](../../vision/blocks/restoration.py)
- [ml/blocks/validation.py](../../ml/blocks/validation.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
