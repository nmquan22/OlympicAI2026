# LAW XX: nhãn nhiễu và annotation

**LAW / ACL 2026 · workshop · ưu tiên B · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

LAW XX là workshop annotation tại ACL 2026. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://sigann.github.io/LAW-XX-2026/).

## Đề luyện do repo đề xuất

Dữ liệu gán nhãn chưa chắc chắn → dự đoán và danh sách cần audit.

Metric để luyện: **Macro-F1 trên dev sạch; báo disagreement theo annotator nếu có.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Linear text model và rà confusion matrix. |
| Nâng cấp vừa sức | Out-of-fold confidence; giảm trọng số nhãn đáng ngờ sau kiểm tra. |
| Hướng nghiên cứu để đọc tiếp | Đọc human-in-the-loop và uncertainty-aware supervision. |

Phép thử quyết định bước tiếp theo: **Thêm nhiễu nhãn có kiểm soát trên train; không sửa dev để tăng điểm.**

## Chuẩn bị và khối code

Đề audit nhãn là bài tự biên soạn từ chủ đề workshop; không gán cho LAW một shared task chưa xác minh.

- [nlp/blocks/retrieval.py](../../nlp/blocks/retrieval.py)
- [ml/blocks/validation.py](../../ml/blocks/validation.py)
- [common/ensemble.py](../../common/ensemble.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
