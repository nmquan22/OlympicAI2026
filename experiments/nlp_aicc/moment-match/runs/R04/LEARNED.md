# R04 · IDF làm alignment có trọng tâm

Accuracy **69.12%** trên cùng 1,059 validation, tăng **3.68 điểm phần trăm** so với R01 (39 câu đúng thêm). Không thêm encoder layer: chỉ tính IDF từ train để giảm đóng góp token phổ biến khi gộp alignment.

Ý tưởng: câu có nhiều từ chức năng giống nhau không đồng nghĩa cùng sự kiện. Trọng số bằng chứng hữu ích hơn việc tăng độ phức tạp kiến trúc trong lượt này. Chưa có nhiều seed nên chưa khẳng định mức tăng ổn định. Best epoch 3; sau đó vẫn overfit.

Thay kế hoạch R07: ghép **token attention R02 + IDF R04**, vì event attention R03 đã giảm điểm. Đây là quyết định từ validation, chưa xem leaderboard.
