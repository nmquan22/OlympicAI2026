# R02 · Marker chưa dạy model cách thực thi

MAE **37.94**, kém R01 **36.26** trên cùng 3,000 holdout. Thêm separator lệnh và marker số không tự tạo ra khả năng tính toán hay mô phỏng control flow. Train loss giảm 0.43→0.34 trong 30 epoch nhưng validation chỉ dao động quanh 38.

Ý tưởng học: special token giúp biểu diễn vai trò, nhưng có thể không tăng nếu model thiếu inductive bias xử lý giá trị số. R03 sẽ kiểm tra nhánh thống kê số; R04 thay pooling. Không suy rộng rằng [SEP] luôn vô ích: đây là kết quả của cụ thể tokenizer, model và bài này.
