# CODI-CRAC: discourse, coreference và thứ tự

**CODI-CRAC / ACL 2026 · workshop · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Workshop chung về discourse và reference/coreference có phiên bản 2026. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://sites.google.com/view/codi-crac2026/schedule).

## Đề luyện do repo đề xuất

Đoạn văn hoặc turns → cạnh liên kết, antecedent hoặc permutation.

Metric để luyện: **Pair-F1 hoặc pairwise ordering accuracy cho bài luyện.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Pair scorer trên sentence features và distance prior. |
| Nâng cấp vừa sức | Decode có ràng buộc, beam search và negative sampling theo document. |
| Hướng nghiên cứu để đọc tiếp | Đọc discourse representations và document-level inference. |

Phép thử quyết định bước tiếp theo: **So dự đoán từng cạnh với decode toàn cục; split theo document.**

## Chuẩn bị và khối code

Dùng subset train/dev và các weights được đề thi cho phép; chuẩn bị dữ liệu trước buổi luyện.

- [nlp/blocks/ordering.py](../../nlp/blocks/ordering.py)
- [nlp/blocks/changepoint.py](../../nlp/blocks/changepoint.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
