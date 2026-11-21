# SemEval 11: logic tách khỏi content bias

**SemEval / ACL 2026 · shared_task · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Task 11 đánh giá syllogism, premise selection và content effect. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://sites.google.com/view/semeval-2026-task-11).

## Đề luyện do repo đề xuất

Premises + conclusion → validity và tập premise liên quan.

Metric để luyện: **Accuracy và chênh lệch giữa nhóm plausible/implausible; dùng official scorer để xếp hạng.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | TF-IDF hoặc encoder classifier, kèm tập phản ví dụ thay entity. |
| Nâng cấp vừa sức | Chuẩn hóa quan hệ logic và kiểm tra premise relevance. |
| Hướng nghiên cứu để đọc tiếp | Đọc neuro-symbolic reasoning; thử symbolic control với grammar đã biết. |

Phép thử quyết định bước tiếp theo: **Đổi tên và nội dung entity nhưng giữ cấu trúc logic; đo consistency.**

## Chuẩn bị và khối code

Dùng subset train/dev và các weights được đề thi cho phép; chuẩn bị dữ liệu trước buổi luyện.

- [nlp/blocks/structured.py](../../nlp/blocks/structured.py)
- [ml/blocks/validation.py](../../ml/blocks/validation.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
