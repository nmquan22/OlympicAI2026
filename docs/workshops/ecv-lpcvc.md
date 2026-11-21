# ECV/LPCVC: retrieval và video với ngân sách inference

**ECV / CVPR 2026 · workshop · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

LPCVC 2026 có track image–text retrieval, video action và AI-generated image detection. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://ecv-workshop.github.io/).

## Đề luyện do repo đề xuất

Embedding hoặc clip feature → rank/lớp; có đo latency.

Metric để luyện: **R@K hoặc accuracy cùng latency của thiết bị luyện.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Cache feature; thử linear head/prototype trước khi fine-tune. |
| Nâng cấp vừa sức | Distill head, giảm chiều embedding và tìm Pareto accuracy–latency. |
| Hướng nghiên cứu để đọc tiếp | Đọc quantization và deployment trên hardware mục tiêu. |

Phép thử quyết định bước tiếp theo: **So single model và ensemble theo cả score lẫn thời gian; warm-up trước khi đo.**

## Chuẩn bị và khối code

CPU/T4 timing không đại diện cho điểm latency trên thiết bị LPCVC chính thức.

- [common/runtime.py](../../common/runtime.py)
- [common/ensemble.py](../../common/ensemble.py)
- [vision_ml/blocks/fusion.py](../../vision_ml/blocks/fusion.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
