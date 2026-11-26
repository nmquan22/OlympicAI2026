# R01 · Transformer không tự mang khả năng thực thi chương trình

Model BPE Transformer khởi tạo mới đạt **MAE 36.2603** trên 3.000 mẫu holdout. Đây chưa tốt bằng MAE 22.6452 trong notebook reference, nhưng khác split/budget nên không phải so sánh có kiểm soát.

Idea được kiểm tra là thay GRU bằng self-attention và tối ưu MAE trực tiếp. Kết quả cho thấy chỉ đổi backbone chưa giải quyết được biểu diễn số và control flow. Các lượt kế tiếp tách riêng separator lệnh, nhánh numeric và query pooling; không khẳng định thêm attention chắc chắn tốt hơn.

Lỗi lưu draft trước đó đã được khôi phục bằng Import Notebook. Version 1 thực sự chạy T4 và có report; các lần lưu lỗi không tính là experiment. Chưa có điểm Kaggle hoặc phân tích theo mẫu.
