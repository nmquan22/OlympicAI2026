# SELVA: distillation và hiệu quả đa phương thức

**SELVA / ACL 2026 · workshop · ưu tiên B · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

SELVA là workshop đầu tiên về hiệu quả language–vision–action. Nguồn xác nhận đây là lần đầu của workshop/challenge tại sự kiện này. [Nguồn chính thức](https://selva-workshop.github.io/).

## Đề luyện do repo đề xuất

Logits/features teacher được cấp → student gọn.

Metric để luyện: **Score đề + milliseconds/sample + peak memory.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Student học supervised bằng CE. |
| Nâng cấp vừa sức | Thêm KL distillation có temperature, cache logits khi augmentation cố định. |
| Hướng nghiên cứu để đọc tiếp | Đọc pruning/quantization chỉ khi đã đo bottleneck. |

Phép thử quyết định bước tiếp theo: **So CE và CE+KL với cùng student, dữ liệu và thời gian.**

## Chuẩn bị và khối code

Teacher chỉ được dùng nếu đề cho phép; teacher/student phải cùng class order.

- [common/runtime.py](../../common/runtime.py)
- [common/ensemble.py](../../common/ensemble.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
