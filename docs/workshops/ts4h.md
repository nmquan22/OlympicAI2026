# TS4H: chuỗi không đều, missingness và subject split

**TS4H / ICLR 2026 · workshop · ưu tiên B · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Workshop 2026 nghiên cứu dữ liệu sức khỏe theo thời gian và đa modality. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://timeseries4health.github.io/iclr2026/index.html).

## Đề luyện do repo đề xuất

Chuỗi cảm biến, timestamp, mask → class hoặc value tương lai.

Metric để luyện: **AUROC/macro-F1 hoặc MAE theo bài luyện; split theo subject.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Mean/std/missingness features + linear/tree model. |
| Nâng cấp vừa sức | Temporal model nhận delta-time, mask và pooling theo độ dài. |
| Hướng nghiên cứu để đọc tiếp | Đọc irregular temporal representations và multimodal alignment. |

Phép thử quyết định bước tiếp theo: **So bỏ mask với thêm mask/delta-time; tuyệt đối không forward-fill từ tương lai.**

## Chuẩn bị và khối code

Bài luyện ML, không nhằm chẩn đoán lâm sàng; chọn dataset công khai đã có consent/license.

- [ml/blocks/timeseries.py](../../ml/blocks/timeseries.py)
- [ml/blocks/tabular.py](../../ml/blocks/tabular.py)
- [ml/blocks/validation.py](../../ml/blocks/validation.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
