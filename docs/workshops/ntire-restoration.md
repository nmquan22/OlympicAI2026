# NTIRE: restoration và đánh giá chất lượng

**NTIRE / CVPR 2026 · workshop · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Phiên bản 2026 có nhiều track restoration, gồm SR x4 và denoising. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://www.cvlai.net/ntire/2026/).

## Đề luyện do repo đề xuất

Ảnh suy giảm → ảnh sạch cùng hệ tọa độ.

Metric để luyện: **PSNR theo ảnh; cố định RGB/Y, data range và crop biên.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Bicubic hoặc identity, rồi residual CNN với L1/Charbonnier. |
| Nâng cấp vừa sức | Huấn luyện crop nhỏ; thử thêm edge loss với trọng số nhỏ và suy luận có halo. |
| Hướng nghiên cứu để đọc tiếp | Đọc hướng phục hồi sinh ảnh khi đề cho weights; đánh giá cả tính trung thực và cảm nhận. |

Phép thử quyết định bước tiếp theo: **So L1 và L1+edge trên cùng scene split; kiểm tra chi tiết có bị bịa thêm.**

## Chuẩn bị và khối code

Dùng subset train/dev và các weights được đề thi cho phép; chuẩn bị dữ liệu trước buổi luyện.

- [vision/blocks/restoration.py](../../vision/blocks/restoration.py)
- [vision/restore.py](../../vision/restore.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
