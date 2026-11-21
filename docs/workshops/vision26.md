# VISION: kiểm tra công nghiệp và ảnh X-ray

**VISION / CVPR 2026 · workshop · ưu tiên B · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Workshop 2026 có các track General, DEXTER và INSIDE-X. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://vision-workshop-26.github.io/).

## Đề luyện do repo đề xuất

Ảnh kiểm tra → lớp lỗi, vùng lỗi hoặc ảnh phục hồi tùy track.

Metric để luyện: **Macro-F1/IoU hoặc PSNR tùy bài luyện; không trộn ba nhiệm vụ.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Dùng classifier/segmenter rẻ để biết lỗi nằm ở ảnh hay nhãn. |
| Nâng cấp vừa sức | Kết hợp patch descriptor với metadata quy trình khi có. |
| Hướng nghiên cứu để đọc tiếp | Đọc physics-informed reconstruction và open-set inspection. |

Phép thử quyết định bước tiếp theo: **So random split với split theo sản phẩm/lô; đo khoảng cách generalization.**

## Chuẩn bị và khối code

Dùng subset train/dev và các weights được đề thi cho phép; chuẩn bị dữ liệu trước buổi luyện.

- [vision_ml/blocks/anomaly.py](../../vision_ml/blocks/anomaly.py)
- [ml/blocks/tabular.py](../../ml/blocks/tabular.py)
- [vision/blocks/restoration.py](../../vision/blocks/restoration.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
