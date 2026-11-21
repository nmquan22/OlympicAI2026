# DCASE 6: tìm đoạn trong audio dài

**DCASE 2026 · challenge · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Task 6 cung cấp cached audio–text features và xếp hạng chính bằng recall1@0.7. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://dcase.community/challenge2026/task-audio-moment-retrieval-from-long-audio).

## Đề luyện do repo đề xuất

Query + clip embeddings/timestamps → một interval.

Metric để luyện: **Recall@1 tại temporal IoU 0.7; giữ scorer gốc khi so chính thức.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Cosine clip/query, center score và tìm interval theo giới hạn độ dài. |
| Nâng cấp vừa sức | TemporalCNN/context smoothing; tune length penalty trên dev. |
| Hướng nghiên cứu để đọc tiếp | Đọc temporal cross-attention và multi-scale localization. |

Phép thử quyết định bước tiếp theo: **So frame maximum, interval sum và length-penalized decoder.**

## Chuẩn bị và khối code

Ưu tiên cached features từ link organizer để tránh phụ thuộc YouTube; baseline repo chỉ decoder, chưa có official-format adapter.

- [audio/blocks/temporal.py](../../audio/blocks/temporal.py)
- [ml/blocks/timeseries.py](../../ml/blocks/timeseries.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
