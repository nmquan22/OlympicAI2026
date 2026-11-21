# DCASE 2: anomalous sound, nhiễu và domain shift

**DCASE 2026 · challenge · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Task 2 năm 2026 nhấn mạnh noise-aware machine anomaly. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring).

## Đề luyện do repo đề xuất

Âm thanh máy + machine/section → anomaly score.

Metric để luyện: **AUROC/pAUC theo machine/condition của bài luyện.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Log-mel statistics rồi nearest-neighbor normal memory. |
| Nâng cấp vừa sức | Separate section statistics, robust scaling và pooled encoder features. |
| Hướng nghiên cứu để đọc tiếp | Đọc noise-invariant representations và domain-aware anomaly scoring. |

Phép thử quyết định bước tiếp theo: **Thêm noise ở dev có kiểm soát; so patch/audio-global descriptor cùng memory budget.**

## Chuẩn bị và khối code

Chỉ fit bằng normal train; khối patch memory cần adapter feature audio, không đọc test labels.

- [audio/blocks/features.py](../../audio/blocks/features.py)
- [ml/blocks/timeseries.py](../../ml/blocks/timeseries.py)
- [vision_ml/blocks/anomaly.py](../../vision_ml/blocks/anomaly.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
