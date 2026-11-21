# DCASE 4: tách và phân đoạn sound scene

**DCASE 2026 · challenge · ưu tiên C · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Task 4 có spatial sound scene segmentation, gồm target có thể vắng mặt. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://dcase.community/challenge2026/task-spatial-semantic-segmentation-of-sound-scenes).

## Đề luyện do repo đề xuất

Mixture đa kênh → tín hiệu nguồn và metadata.

Metric để luyện: **Scorer tách nguồn/spatial chính thức; không dùng F1 đơn lẻ thay thế.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Bài rút gọn: mask STFT cho một target đã biết. |
| Nâng cấp vừa sức | Thêm presence head để tránh sinh nguồn khi target vắng. |
| Hướng nghiên cứu để đọc tiếp | Đọc spatial source separation và object-centric audio. |

Phép thử quyết định bước tiếp theo: **Thử absent target và hai nguồn cùng lớp trước khi tăng backbone.**

## Chuẩn bị và khối code

Chưa có trainer separation trong repo; là hướng mở rộng C, cần thêm multi-channel loader và inverse STFT.

- [audio/blocks/features.py](../../audio/blocks/features.py)
- [vision/blocks/models.py](../../vision/blocks/models.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
