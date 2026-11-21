# GenRecon3D: hình học và độ bất định

**GenRecon3D / CVPR 2026 · workshop · ưu tiên C · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Workshop mới về reconstruction gắn với generative modeling. Nguồn xác nhận đây là lần đầu của workshop/challenge tại sự kiện này. [Nguồn chính thức](https://genrecon3d.github.io/).

## Đề luyện do repo đề xuất

Quan sát thiếu → occupancy/geometry và độ bất định.

Metric để luyện: **IoU/Chamfer của bài luyện; đo hình học thay vì chỉ ảnh render.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Coordinate MLP hoặc occupancy head trên feature được cấp. |
| Nâng cấp vừa sức | Fourier encoding và sampling tập trung vùng biên. |
| Hướng nghiên cứu để đọc tiếp | Đọc generative geometry; full 3D diffusion nằm ngoài baseline 5 tiếng. |

Phép thử quyết định bước tiếp theo: **So positional encoding với tăng width ở cùng số tham số.**

## Chuẩn bị và khối code

Dùng bài coordinate regression/occupancy nhỏ; không giả định tái lập hệ thống 3D lớn trong 5 tiếng.

- [vision/blocks/fields.py](../../vision/blocks/fields.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
