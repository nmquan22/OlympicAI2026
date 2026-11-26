# R05 · Làm soft alignment sắc hơn

Temperature 0.20→0.07 đạt **66.10%**, tăng 0.66 điểm phần trăm so với R01 nhưng kém IDF R04 69.12%. Cùng train/validation, seed, backbone và optimizer; thay độ tập trung của trọng số matching.

Temperature thấp làm softmax gần phép max hơn, nhấn mạnh một token khớp nhất. Có ích nhẹ ở lượt này nhưng không xử lý được token phổ biến giống IDF. Chưa kết luận mức tăng ổn định khi chưa chạy nhiều seed. Giữ R04 là ứng viên tốt nhất hiện tại.
