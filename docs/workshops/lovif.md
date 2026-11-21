# LoViF: restoration với reward và mô hình sinh

**LoViF / CVPR 2026 · workshop · ưu tiên C · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

LoViF là workshop đầu tiên, kết hợp low-level vision, preference và agentic systems. Nguồn xác nhận đây là lần đầu của workshop/challenge tại sự kiện này. [Nguồn chính thức](https://lovif-cvpr2026-workshop.github.io/).

## Đề luyện do repo đề xuất

Ảnh suy giảm + điều kiện → ảnh phục hồi.

Metric để luyện: **PSNR và đánh giá cảm nhận của bài luyện; tách hai mục tiêu.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Residual CNN dùng loss trực tiếp trên pixel. |
| Nâng cấp vừa sức | Thử residual conditioning, edge loss và chọn checkpoint bằng metric đề. |
| Hướng nghiên cứu để đọc tiếp | Đọc preference optimization và conditional generative restoration. |

Phép thử quyết định bước tiếp theo: **Giữ cùng ảnh; so độ sắc với lỗi pixel và artifact do sinh ảnh.**

## Chuẩn bị và khối code

Chỉ thử mô hình sinh khi đã có weights và pipeline tương thích offline.

- [vision/blocks/restoration.py](../../vision/blocks/restoration.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
