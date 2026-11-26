# AICC experiment agent

Đây là quy trình cho trợ lý có trình duyệt Kaggle và quyền ghi repo; chưa phải daemon hoặc một dịch vụ chạy liên tục. Hàng đợi nằm trong `queue.json`. Chỉ báo hoàn thành khi đã đọc được log/output thực tế của version Kaggle.

## Luồng một bài

1. Đọc đề, evaluator, giới hạn dữ liệu/model, baseline và nguồn code công khai. Dẫn nguồn và giữ giấy phép nếu sao chép code.
2. Kiểm tra Active Events. Tổng phiên đang chạy, tính cả phiên người dùng đã mở, không vượt 2. Không dừng phiên khác. Hết chỗ thì giữ bài ở queued.
3. Chọn validation đúng đơn vị tổng quát hóa: race/biome, document, patient, video hoặc domain. Không dùng test để chọn cấu hình.
4. Chạy baseline trên dữ liệu thật. Thử 1–2 thay đổi có giả thuyết cụ thể, cùng split/seed. Lưu ablation giảm điểm như bình thường.
5. Save & Run All. Phân biệt queued, running, failed và completed. Không suy ra chạy thành công từ một draft hoặc thao tác bấm Save.
6. Đọc log, kiểm tra output, ID/số dòng/schema và điều kiện riêng của metric. Giữ raw score và cách chuẩn hóa. Score validation và score Kaggle là hai trường riêng.
7. Ghi notebook, config, số liệu, thời gian, phiên bản dữ liệu, nhận xét và bước kế tiếp vào repo. Chỉ ghi số đã quan sát; chưa có leaderboard thì để null.
8. Dừng một bài khi baseline tái lập được, có validation hợp lệ, ablation đủ quyết định và output hợp lệ; hoặc ghi rõ lỗi/chặn còn lại. Nhả slot trước khi chuyển bài tiếp.

## Đầu ra

Mỗi bài có thư mục riêng với notebook và báo cáo; metric dạng JSON/CSV để dễ tổng hợp. Không commit dataset gốc, checkpoint lớn, thông tin đăng nhập, cookie hoặc API key. Những bài cần checkpoint có sẵn phải ghi rõ nguồn và điều kiện đề.

Kaggle có thể yêu cầu chấp nhận quy định từng cuộc thi. Nếu chưa có xác nhận, giữ trạng thái awaiting_rules; không bấm chấp nhận thay người dùng. Cosmos đã được người dùng xác nhận cho mục đích luyện tập sau cuộc thi.

Smoke test hiện tại chỉ dành cho Cosmos. Chín mục còn lại là hàng đợi, chưa có thí nghiệm hay điểm số. Việc tiếp tục chạy phụ thuộc phiên trợ lý/trình duyệt còn hoạt động; chưa cài lịch chạy nền.

