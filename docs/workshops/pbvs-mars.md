# PBVS: phân đoạn sạt lở trên Sao Hỏa

**PBVS / CVPR 2026 · challenge · ưu tiên B · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

MARS-LS là challenge mới về phân đoạn sạt lở. Nguồn xác nhận đây là lần đầu của workshop/challenge tại sự kiện này. [Nguồn chính thức](https://pbvs-workshop.github.io/challenge.html).

## Đề luyện do repo đề xuất

Ảnh địa hình và các modality được cấp → mask sạt lở.

Metric để luyện: **Foreground IoU, Dice và lỗi theo scene cho bài luyện.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | U-Net trên từng modality; kiểm tra mask và vùng ignore. |
| Nâng cấp vừa sức | Fuse thông tin hình thái/địa hình khi được cấp; dùng crop có đủ positive. |
| Hướng nghiên cứu để đọc tiếp | Đọc domain adaptation địa hình và multimodal segmentation. |

Phép thử quyết định bước tiếp theo: **So RGB-only với thêm địa hình; chia dev theo vùng thay vì patch gần nhau.**

## Chuẩn bị và khối code

Loader/mask theo track PBVS; không giả định dữ liệu có DEM nếu đề không cấp.

- [vision/blocks/models.py](../../vision/blocks/models.py)
- [vision/blocks/masks.py](../../vision/blocks/masks.py)
- [vision_ml/blocks/multimodal.py](../../vision_ml/blocks/multimodal.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
