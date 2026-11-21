# DataCV: chọn dữ liệu và chất lượng nhãn

**DataCV / CVPR 2026 · workshop · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Phiên bản 2026 mở rộng data-centric vision sang dữ liệu đa phương thức. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://sites.google.com/view/datacv-2026-cvpr/).

## Đề luyện do repo đề xuất

Training pool + budget → subset hoặc sample weights.

Metric để luyện: **Accuracy/macro-F1 khi cố định số mẫu và số bước train.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Random và class-balanced subsets làm hai mốc. |
| Nâng cấp vừa sức | Thử chọn đa dạng trên embedding, loại near-duplicate sau khi audit. |
| Hướng nghiên cứu để đọc tiếp | Đọc data valuation và selection; recipe này là bài tự xây từ chủ đề workshop. |

Phép thử quyết định bước tiếp theo: **Giữ nguyên model và compute; so random/diversity/hard mining trên 3 seed.**

## Chuẩn bị và khối code

Workshop theme đã xác minh; bài subset selection ở đây là đề luyện tự biên soạn, không phải mô tả challenge chính thức.

- [vision/blocks/self_supervised.py](../../vision/blocks/self_supervised.py)
- [ml/blocks/validation.py](../../ml/blocks/validation.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
