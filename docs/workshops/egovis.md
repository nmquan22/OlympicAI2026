# EgoVis: tìm hành động trong video bằng câu mô tả

**EgoVis / CVPR 2026 · workshop · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Workshop bao phủ egocentric video và có các challenge định vị thời gian. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://egovis.github.io/cvpr26/).

## Đề luyện do repo đề xuất

Text query + sequence clip features → khoảng start/end.

Metric để luyện: **R@1 với temporal IoU 0.5/0.7 cho bài luyện.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Cosine giữa query và từng clip; chọn interval có tổng score đã center. |
| Nâng cấp vừa sức | Thêm context bằng temporal CNN hoặc score smoothing. |
| Hướng nghiên cứu để đọc tiếp | Đọc grounding nhiều tỉ lệ thời gian và hard negatives. |

Phép thử quyết định bước tiếp theo: **So pooling toàn video với clip localization; chia split theo video/người quay.**

## Chuẩn bị và khối code

Dùng subset train/dev và các weights được đề thi cho phép; chuẩn bị dữ liệu trước buổi luyện.

- [audio/blocks/temporal.py](../../audio/blocks/temporal.py)
- [ml/blocks/timeseries.py](../../ml/blocks/timeseries.py)
- [vision_ml/blocks/fusion.py](../../vision_ml/blocks/fusion.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
