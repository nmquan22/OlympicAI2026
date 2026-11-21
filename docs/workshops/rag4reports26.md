# RAG4Reports: báo cáo có dẫn evidence

**RAG4Reports / ACL 2026 · workshop · ưu tiên B · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

RAG4Reports là workshop mới về báo cáo đa ngôn ngữ bằng RAG. Nguồn xác nhận đây là lần đầu của workshop/challenge tại sự kiện này. [Nguồn chính thức](https://2026.aclweb.org/program/workshops/).

## Đề luyện do repo đề xuất

Nhu cầu thông tin + corpus → evidence và báo cáo.

Metric để luyện: **Evidence Recall@K và coverage của bài luyện.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Retrieve rồi extractive summary có passage ID. |
| Nâng cấp vừa sức | Diversify evidence và kiểm tra coverage câu hỏi con. |
| Hướng nghiên cứu để đọc tiếp | Đọc multilingual long-form grounded generation. |

Phép thử quyết định bước tiếp theo: **Tắt generator, dùng extractive control để biết lỗi đến từ retrieval hay diễn đạt.**

## Chuẩn bị và khối code

Chỉ phần retrieval chạy từ khối code sẵn; generator cần model do đề cấp.

- [nlp/blocks/retrieval.py](../../nlp/blocks/retrieval.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
