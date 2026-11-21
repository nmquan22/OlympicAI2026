# TrustNLP: robustness và confidence

**TrustNLP / ACL 2026 · workshop · ưu tiên B · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

TrustNLP 2026 bao phủ robustness và các thuộc tính đáng tin cậy của NLP. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://2026.aclweb.org/program/workshops/).

## Đề luyện do repo đề xuất

Text theo domain/nhóm → nhãn, confidence.

Metric để luyện: **Macro-F1, worst-group accuracy, NLL cho bài luyện.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Text classifier và calibration split riêng. |
| Nâng cấp vừa sức | Temperature scaling, perturbation test không đổi nhãn. |
| Hướng nghiên cứu để đọc tiếp | Đọc calibration dưới domain shift và selective prediction. |

Phép thử quyết định bước tiếp theo: **So threshold chung với từng domain; không fit temperature trên test.**

## Chuẩn bị và khối code

Đề robustness là bài tự xây; workshop listing không xác nhận một benchmark duy nhất.

- [common/ensemble.py](../../common/ensemble.py)
- [ml/blocks/validation.py](../../ml/blocks/validation.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
