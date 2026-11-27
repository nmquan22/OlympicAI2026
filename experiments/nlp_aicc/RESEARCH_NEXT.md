# Research tiếp sau bảy lượt

## Moment Match: từ độ giống cục bộ tới thứ tự sự kiện

Bằng chứng: R04 IDF 69.12%, R06 chỉ target center 49.10%, R03 broadcast event context 60.43%, R07 token attention+IDF 66.48%. Do đó giữ chứng cứ lân cận và IDF, thử prior thứ tự nhẹ hơn một Transformer mới.

R08 thêm một DP lên grid 5×9. Một đường chọn năm candidate event có chỉ số tăng dần. Thay max đường bằng log-sum-exp để gradient truyền qua các đường. Điểm CNN cộng một hệ số học được nhân điểm alignment. Không bắt từng từ khớp cứng, không cần pretrained mới. Có thể thất bại nếu nhãn chứa thứ tự khác hoặc nhiều sự kiện bị lược.

\[D_{i,j}=S_{i,j}+\tau\log\sum_{k<j}\exp(D_{i-1,k}/\tau).\]

Chuẩn hóa số đường và độ dài để điểm prior có thang đo ổn định. Đây là **strict subsequence DP**, không gọi là triển khai đầy đủ Soft-DTW. T4 canary đối chiếu mười đường vét cạn và kiểm tra gradient hữu hạn trước khi train. R09 riêng biệt thử max token alignment như ý tưởng trong reference, giữ IDF và BPE để so với R04.

Nguồn đã đọc: [Differentiable Dynamic Programming, ICML 2018](https://proceedings.mlr.press/v80/mensch18a.html), [Soft-DTW, ICML 2017](https://proceedings.mlr.press/v70/cuturi17a.html). Đây là chuyển ý tưởng sang bài mới, không phải bằng chứng điểm sẽ tăng trên Moment Match.

## Lost Interpreter: giữ giá trị số đúng vị trí

R02–R05 chưa vượt R01 MAE 36.26. Global numeric summaries làm mất thứ tự; query pooling và đếm số lệnh chưa cung cấp trạng thái thực thi.

Sau R06/R07, nếu vẫn không cải thiện: so BPE BiGRU được sửa padding/masking, SmoothL1 và full-sequence state với scratch Transformer trong **cùng holdout**; tiếp theo thử value channel ở từng instruction thay vì chỉ tổng số toàn chương trình. Cần đọc ví dụ train thật trước khi đóng parser/opcode assumptions.

[Neural Arithmetic Units, ICLR 2020](https://arxiv.org/abs/2001.05016) chỉ ra vai trò của bias phép toán và điều kiện tối ưu hóa. Không khẳng định một NAU tự học interpreter có nhảy lệnh; chưa triển khai model đó khi chưa xác nhận grammar dữ liệu.

## Chờ GPU

Giữ tối đa hai Saved Version. Batch còn lại Word Lookups dự kiến khoảng 14 phút, không mở trạng thái từng phút. Inverse R01 dùng canary bốn ví dụ và Transformers 4.57.1; chỉ kiểm tra sau khoảng 4–5 phút. Tách lỗi cài package, lỗi canary, runtime train và thời gian queue. Thí nghiệm lỗi không tính vào số đã hoàn tất.

R08 Lost Interpreter đã chuẩn bị code: BiGRU train từ đầu với packed sequences, mean/max pooling có mask, SmoothL1 beta 0.2. So sánh cả một ứng viên (backbone + loss), không gọi là ablation chỉ đổi backbone. Thêm log chương trình train và nhóm lỗi theo độ dài để quyết định thử value channel; chưa có kết quả R08.
