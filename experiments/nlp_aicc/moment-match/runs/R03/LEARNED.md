# R03 · Attention ở mức sự kiện

- T4 thực chạy: accuracy **60.43%** trên cùng 1,059 validation; R01 65.44%, R02 66.38%.
- Thay đổi so với R01: thêm positional event Transformer và broadcast context vào token, giữ BPE và soft late interaction.
- Best epoch 4, sau đó train loss tiếp tục giảm từ 0.77 xuống 0.37 nhưng validation không tăng: dấu hiệu overfit. Chưa đủ bằng chứng nói event attention luôn kém; kết quả chỉ bác bỏ cách ghép và ngân sách hiện tại.
- Idea học: late interaction giữ bằng chứng khớp cục bộ; trộn context trước matching có thể làm giảm độ sắc của alignment. Nhánh token attention R02 là ứng viên tốt hơn hiện tại.
- Không dùng leaderboard để chọn cấu hình. Chưa có kiểm định nhiều seed hay phân nhóm lỗi theo từng mẫu.
