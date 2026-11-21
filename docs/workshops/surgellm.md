# SURGeLLM: bảng, schema và truy hồi có cấu trúc

**SURGeLLM / ACL 2026 · workshop · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

Workshop mới nghiên cứu structured understanding, retrieval và generation. Nguồn xác nhận đây là lần đầu của workshop/challenge tại sự kiện này. [Nguồn chính thức](https://surgellm.github.io/acl2026/).

## Đề luyện do repo đề xuất

Câu hỏi + bảng/schema → evidence hoặc truy vấn có cấu trúc.

Metric để luyện: **Evidence Recall@K và execution accuracy cho bài luyện.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Retrieve tên bảng/cột bằng TF-IDF; thực thi template nhỏ. |
| Nâng cấp vừa sức | Schema linking, index cell/row và kiểm tra kiểu output. |
| Hướng nghiên cứu để đọc tiếp | Đọc reasoning nhiều bảng và schema-aware generation. |

Phép thử quyết định bước tiếp theo: **Ẩn tên cột dễ đoán và đổi thứ tự hàng; kiểm tra tính bất biến.**

## Chuẩn bị và khối code

Không thực thi SQL sinh tự do trên dữ liệu ngoài bài; baseline chỉ hướng dẫn retrieval/schema, chưa có text-to-SQL trainer.

- [nlp/blocks/retrieval.py](../../nlp/blocks/retrieval.py)
- [nlp/blocks/structured.py](../../nlp/blocks/structured.py)
- [ml/blocks/tabular.py](../../ml/blocks/tabular.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
