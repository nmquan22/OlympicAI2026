# SemEval 4: tương đồng cốt truyện

**SemEval / ACL 2026 · workshop · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Task 4 tập trung narrative similarity và representation learning. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://narrative-similarity-task.github.io/).

## Đề luyện do repo đề xuất

Cặp truyện → score tương đồng hoặc thứ hạng.

Metric để luyện: **Spearman/nDCG của bài luyện; đối chiếu subtask chính thức.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | TF-IDF và cosine của frozen sentence embedding. |
| Nâng cấp vừa sức | So pair feature với pooling theo đoạn; hard negative cùng nhân vật khác plot. |
| Hướng nghiên cứu để đọc tiếp | Đọc biểu diễn sự kiện và narrative abstraction. |

Phép thử quyết định bước tiếp theo: **Đổi tên nhân vật ở dev; xem model có đang bám tên thay vì quan hệ sự kiện.**

## Chuẩn bị và khối code

Dùng subset train/dev và các weights được đề thi cho phép; chuẩn bị dữ liệu trước buổi luyện.

- [nlp/blocks/retrieval.py](../../nlp/blocks/retrieval.py)
- [nlp/blocks/ordering.py](../../nlp/blocks/ordering.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
