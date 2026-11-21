# SemEval 3: sentiment theo aspect và cường độ

**SemEval / ACL 2026 · workshop · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Task 3 năm 2026 là DimABSA. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://github.com/DimABSA/DimABSA2026).

## Đề luyện do repo đề xuất

Văn bản → aspect/opinion và giá trị sentiment tùy subtask.

Metric để luyện: **Span-F1 và MAE/Pearson cho bài luyện; scorer DimABSA mới là chuẩn challenge.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Encoder + token head để tìm span; head hồi quy riêng cho intensity. |
| Nâng cấp vừa sức | Joint extraction; mask label thiếu và cân bằng các loss. |
| Hướng nghiên cứu để đọc tiếp | Đọc structured prediction không mất alignment token–character. |

Phép thử quyết định bước tiếp theo: **So classification polarity với regression; phân tích lỗi extraction trước intensity.**

## Chuẩn bị và khối code

Dùng subset train/dev và các weights được đề thi cho phép; chuẩn bị dữ liệu trước buổi luyện.

- [nlp/blocks/structured.py](../../nlp/blocks/structured.py)
- [nlp/blocks/transformers.py](../../nlp/blocks/transformers.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
