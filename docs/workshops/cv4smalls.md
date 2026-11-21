# CV4Smalls: video với ít nhãn

**CV4Smalls / CVPR 2026 · workshop · ưu tiên B · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

CV4Smalls 2026 nhấn mạnh dữ liệu nhỏ và video dynamics. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://cvpr.thecvf.com/Conferences/2026/Workshops).

## Đề luyện do repo đề xuất

Ít video có nhãn → lớp hoặc tương đồng hành động.

Metric để luyện: **Accuracy theo video; confidence interval qua split.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Mean-pool frame feature và linear classifier. |
| Nâng cấp vừa sức | TemporalCNN và augmentation temporal; frozen encoder làm control. |
| Hướng nghiên cứu để đọc tiếp | Đọc self-supervised temporal representations. |

Phép thử quyết định bước tiếp theo: **Shuffle frame order ở dev; nếu score giữ nguyên, model chưa dùng dynamics.**

## Chuẩn bị và khối code

Danh mục conference đã xác minh; trang workshop trả lỗi khi truy cập, chưa kiểm chứng track/data riêng.

- [ml/blocks/timeseries.py](../../ml/blocks/timeseries.py)
- [vision/blocks/self_supervised.py](../../vision/blocks/self_supervised.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
