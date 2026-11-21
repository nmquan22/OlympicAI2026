# DCASE 5: audio QA phải nghe mới trả lời được

**DCASE 2026 · challenge · ưu tiên B · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Task 5 là audio-dependent multiple-choice QA. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://dcase.community/challenge2026/task-audio-dependent-question-answering).

## Đề luyện do repo đề xuất

Audio + câu hỏi + đáp án lựa chọn → một lựa chọn.

Metric để luyện: **Accuracy, kèm audio-shuffle/silence control trên bài luyện.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Text-only control và cosine audio–answer trên features được cấp. |
| Nâng cấp vừa sức | Cross-attention audio–question, hard negative đáp án hợp ngữ cảnh nhưng sai âm thanh. |
| Hướng nghiên cứu để đọc tiếp | Đọc audio-grounded reasoning và kiểm soát text shortcut. |

Phép thử quyết định bước tiếp theo: **Thay audio bằng silence và shuffle audio trong batch; score phải giảm hợp lý.**

## Chuẩn bị và khối code

SpatialQuestionHead nhận được audio clip regions; đây là adapter ý tưởng, không phải complete LALM.

- [audio/blocks/models.py](../../audio/blocks/models.py)
- [vision_ml/blocks/fusion.py](../../vision_ml/blocks/fusion.py)
- [vision_ml/blocks/multimodal.py](../../vision_ml/blocks/multimodal.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
