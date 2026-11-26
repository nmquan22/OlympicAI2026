# AICC · Nhật ký thí nghiệm

Nguồn lựa chọn: [IOAI Training](https://ioai.artix.tech/problems), bộ lọc AICC, 36 bài tại thời điểm kiểm tra. Chọn 10 bài mới nhất để lập hàng đợi; smoke test Cosmos trước khi triển khai cả loạt.

- [Quy trình agent](AGENT.md)
- [Hàng đợi và trạng thái thật](queue.json)
- [Cosmos Cosmic Run](cosmos-cosmic-run/REPORT.md)

Giới hạn: tối đa 2 phiên đang chạy đồng thời, bao gồm các phiên có sẵn của người dùng. Notebook thường để private; báo cáo/code được lưu vào repo theo yêu cầu người dùng. Không tự công bố notebook hoặc suy diễn điểm leaderboard từ điểm validation.

Mỗi lần ghi nhận một run đã hoàn tất, dùng `python tools/aicc_record_run.py --report <smoke_report.json> --challenge <id> --notebook-url <url> --version <n>`. Công cụ chỉ ghi nhận bằng chứng của run, không tự chạy Kaggle hoặc tự sinh điểm.


Công cụ ghi nhận hiện hỗ trợ schema báo cáo Cosmos (Kendall tau và kiểm tra hoán vị). Những bài dùng metric/output khác cần adapter riêng khi triển khai.
