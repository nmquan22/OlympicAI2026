# BEA: chấm điểm bài viết và feedback

**BEA / ACL 2026 · workshop · ưu tiên B · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

BEA 2026 có trong chương trình ACL. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://2026.aclweb.org/program/workshops/).

## Đề luyện do repo đề xuất

Essay + prompt/rubric → ordinal score trong bài luyện.

Metric để luyện: **Quadratic weighted kappa/MAE của bài tự xây.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | TF-IDF + ridge; thử length-only làm control. |
| Nâng cấp vừa sức | Encoder frozen + ordinal head, split theo prompt hoặc người viết. |
| Hướng nghiên cứu để đọc tiếp | Đọc rubric-conditioned scoring và feedback grounding. |

Phép thử quyết định bước tiếp theo: **So random split với held-out prompt; tránh học độ dài bài như nhãn.**

## Chuẩn bị và khối code

Chỉ xác minh workshop; không khẳng định BEA 2026 có shared task chấm essay cụ thể.

- [ml/blocks/tabular.py](../../ml/blocks/tabular.py)
- [nlp/blocks/transformers.py](../../nlp/blocks/transformers.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
