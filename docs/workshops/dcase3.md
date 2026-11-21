# DCASE 3: semantic acoustic imaging

**DCASE 2026 · challenge · ưu tiên C · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Task 3 dự đoán acoustic field từ spatial audio và audiovisual scenes. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://dcase.community/challenge2026/task-semantic-acoustic-imaging-for-sound-event-localization-and-detection-from-spatial-audio-and-audiovisual-scenes).

## Đề luyện do repo đề xuất

Audio nhiều kênh + hình ảnh theo track → map không gian/thời gian.

Metric để luyện: **Dùng evaluator chính thức; không thay bằng clip accuracy.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Giữ channels, tạo STFT và học map regression nhỏ. |
| Nâng cấp vừa sức | Thêm channel phase/cross-correlation và fusion ảnh khi được cấp. |
| Hướng nghiên cứu để đọc tiếp | Đọc spatial audio representations và dense acoustic fields. |

Phép thử quyết định bước tiếp theo: **So multi-channel với mono control để thấy mất spatial cues.**

## Chuẩn bị và khối code

Chỉ có frontend/model blocks tham khảo; read_wav hiện downmix mono, cần loader multichannel riêng cho track này.

- [audio/blocks/features.py](../../audio/blocks/features.py)
- [vision/blocks/models.py](../../vision/blocks/models.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
