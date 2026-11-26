# R02 · Contextualize token trước khi ghép hai chuỗi

Thêm một Transformer layer có vị trí ở mức BPE token đạt **accuracy 0.663834**, so với **0.654391** ở R01 trên cùng 1.059 mẫu validation: tăng khoảng **0.94 điểm phần trăm** (10 mẫu).

Cơ chế: token trong một câu được nhìn ngữ cảnh của nhau trước khi tính late interaction giữa các sự kiện. Đây là thay đổi có đo được, nhưng chưa kiểm tra nhiều seed nên chưa kết luận có ý nghĩa thống kê. Vẫn thấp hơn số reference 0.7243 và chưa có leaderboard score.

Lượt kế tiếp chuyển phần contextualization lên chuỗi sự kiện để kiểm tra thứ tự và các câu lân cận quan trọng hơn token context tới mức nào.
