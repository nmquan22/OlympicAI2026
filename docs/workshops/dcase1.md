# DCASE 1: phân loại âm thanh theo taxonomy

**DCASE 2026 · challenge · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Task 1 dùng nhãn có cấu trúc phân cấp và audio không đồng nhất. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://dcase.community/challenge2026/task-heterogeneous-audio-classification).

## Đề luyện do repo đề xuất

WAV biến độ dài → sound class ở các cấp.

Metric để luyện: **Macro-F1 để debug; metric phân cấp chính thức theo trang nguồn.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Log-mel CNN với masked temporal pooling. |
| Nâng cấp vừa sức | Crop nhiều đoạn, average clip predictions và thêm head theo taxonomy. |
| Hướng nghiên cứu để đọc tiếp | Đọc pretrained audio encoder nếu đề cấp weights. |

Phép thử quyết định bước tiếp theo: **So single crop và multi-crop, báo nhóm short/long audio.**

## Chuẩn bị và khối code

CSV có id,path,label integer. Baseline một head là bài rút gọn, chưa tái lập taxonomy scorer.

- [audio/classify.py](../../audio/classify.py)
- [audio/blocks/features.py](../../audio/blocks/features.py)
- [audio/blocks/models.py](../../audio/blocks/models.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
