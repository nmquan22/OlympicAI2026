# DCASE 7: continual learning không replay dữ liệu cũ

**DCASE 2026 · challenge · ưu tiên B · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Task 7 yêu cầu học domain nối tiếp; không được truy cập dữ liệu các domain trước. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://dcase.community/challenge2026/task-domain-agnostic-incremental-learning-for-audio-classification).

## Đề luyện do repo đề xuất

Domain mới + model hiện tại → model giữ kiến thức các domain.

Metric để luyện: **Accuracy theo domain và forgetting trong bài luyện.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Fine-tune head trên domain hiện tại, giữ checkpoint trước đó làm control. |
| Nâng cấp vừa sức | Teacher cũ tạo logits trên dữ liệu domain mới để distillation, theo đúng rule track. |
| Hướng nghiên cứu để đọc tiếp | Đọc regularization và parameter isolation; không dùng replay mặc định. |

Phép thử quyết định bước tiếp theo: **Bảng D1/D2/D3 trước–sau mỗi stage; kiểm tra average không che forgetting.**

## Chuẩn bị và khối code

Trang task cấm external data và pretrained models/embeddings ngoài thiết lập organizer. Không dùng replay memory của audio cũ.

- [audio/blocks/models.py](../../audio/blocks/models.py)
- [common/runtime.py](../../common/runtime.py)
- [ml/blocks/validation.py](../../ml/blocks/validation.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
