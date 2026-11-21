# PBVS: EO–SAR, thiếu modality và OOD

**PBVS / CVPR 2026 · workshop · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

MAVIC-C nghiên cứu nhận dạng từ EO/SAR và đánh giá OOD. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://pbvs-workshop.github.io/challenge.html).

## Đề luyện do repo đề xuất

EO, SAR và cờ modality có sẵn → lớp + confidence.

Metric để luyện: **Macro-F1, AUROC OOD trên bài luyện; tách in/out-distribution.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Train hai linear head trên cached feature; late fusion xác suất. |
| Nâng cấp vừa sức | MaskedFusion, modality dropout và calibration trên dev. |
| Hướng nghiên cứu để đọc tiếp | Đọc cross-modal transfer khi chỉ một sensor có mặt ở inference. |

Phép thử quyết định bước tiếp theo: **So EO-only, SAR-only, fusion và che một modality; báo worst condition.**

## Chuẩn bị và khối code

Codabench 12529 được trang PBVS liên kết; preprocessing SAR cần kiểm tra phân bố và dynamic range.

- [vision_ml/blocks/multimodal.py](../../vision_ml/blocks/multimodal.py)
- [common/ensemble.py](../../common/ensemble.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
