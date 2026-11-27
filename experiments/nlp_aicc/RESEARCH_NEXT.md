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


## Alchemy R08: symmetry trong kiến trúc thay vì penalty

Sau bảy lượt, hard negatives tốt nhất 4/30 so với 3/30 ban đầu. Chênh một câu chưa đủ vững. R08 dùng shared BPE embedding từ đầu, mean pool riêng item1/item2/result qua SEP, rồi ghép `[a+b, abs(a-b), a*b]`. Hai thứ tự có cùng representation theo cấu trúc; query và candidate được chấm cosine với scale học được, full-list CE và Hungarian giữ nguyên.

Đây là ứng viên data-efficient để kiểm tra, không phải một SOTA được dẫn từ paper. Tính đối xứng được đảm bảo ở eval, còn semantic generalization vẫn phụ thuộc dữ liệu 120 train rules; có thể vẫn rất thấp. Không tải pretrained BERT để tăng điểm ngoài điều kiện của người học.


### Source preflight

Lượt gọi Lost R08 dùng nhầm source SHA cũ, nên R06/R07 hoàn tất nhưng R08 gặp KeyError trước khi train. Lỗi ở loader đã chuẩn bị, không phải model. Đổi generator nhận SHA bằng tham số tường minh, kiểm tra SHA cụ thể trong text và validate toàn bộ config IDs trước khi chạy lượt đầu. Các lượt research lỗi không tính thêm. Chạy lại batch sạch R06–R08 để có một version hoàn tất đầy đủ.


## Moment R10: giữ nguyên thông tin word vectors được cấp

R08 đạt68.93%, R09 max BPE alignment67.71%, đều chưa vượt IDF69.12%. Đọc lại reference cho thấy họ pad thêm một hàng 0 vào embedding matrix rồi shift word IDs +1; word vectors trong data bắt đầu ở index0 thật. R10 giữ tokenizer BPE, thêm một nhánh **feature** từ nguyên word vectors được đề cấp (frozen) → projection → max late interaction. Ghép hai grid BPE + word ở mức feature map, CNN nhận bốn channel. Không tải model/tokenizer bên ngoài.

Giả thuyết: trung bình vector của nhiều từ vào cùng BPE piece làm nhòe nghĩa, nhất là noun/action hiếm. Đây là giả thuyết được kiểm tra, chưa có điểm R10. Tokenizer của model vẫn là BPE; nhánh phụ là lookup feature từ tài nguyên bài cấp. Điều kiện khác với reference word-only nên không gọi là tái lập exact SOTA.

Alchemy R08 đạt3/30 dù symmetry gap0. R09 thử tied role space và query a+b+residual nhỏ để chuyển thông tin của các tên đã xuất hiện ở input sang result role. Đây là giả thuyết composition từ train rules, không thêm world facts hoặc pretrained.
