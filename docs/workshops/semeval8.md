# SemEval 8: retrieval hội thoại nhiều lượt

**SemEval / ACL 2026 · shared_task · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

MTRAGEval có retrieval-only và hai dạng generation; retrieval được xếp hạng bằng nDCG. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/).

## Đề luyện do repo đề xuất

History + câu hỏi cuối + corpus → top-10 evidence.

Metric để luyện: **nDCG@10; generation dùng scorer khác của ban tổ chức.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | TF-IDF query bằng câu hỏi cuối. |
| Nâng cấp vừa sức | RRF query cuối với query thêm history được chọn; rerank top-k. |
| Hướng nghiên cứu để đọc tiếp | Đọc contextual query rewriting và verification evidence. |

Phép thử quyết định bước tiếp theo: **So last-turn, full-history, selected-history trên cùng corpus; không dùng metadata ẩn ở test.**

## Chuẩn bị và khối code

Ưu tiên Task A cho offline 5 tiếng; generation và LLM judge không có trong baseline repo.

- [nlp/blocks/retrieval.py](../../nlp/blocks/retrieval.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
