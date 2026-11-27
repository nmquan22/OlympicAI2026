# Những attempt không tính thêm

| Attempt | Điều xảy ra | Cách xử lý |
|---|---|---|
| Inverse V01 | Model được cấp không khớp bốn ví dụ dưới phiên bản Transformers mặc định | Giữ canary, tìm phiên bản tương thích |
| Inverse V02 | tokenizers0.19.1 thiếu wheel phù hợp Python3.13, build thất bại | Dùng Transformers4.57.1/tokenizers0.22.2 đã thực chạy |
| Inverse V03 | Search đạt100 nhưng cell cũ tiếp tục cài package rồi lỗi | Xóa cell hoàn toàn, kiểm tra text, chạy lại R01 trong batch sạch V04 |
| Lost V06 | R06/R07 xong, gọi R08 bằng source SHA cũ gây KeyError8 | Source SHA tường minh, preflight toàn config trước train; chỉ tính kết quả sạch V07 |
| Moment V08 | Gọi config8 bằng source SHA chưa chứa nó | Source tường minh và config preflight; chạy sạch V09 |
| Word V02 | Sau sáu config mới, cell cũ gọi lặp R01 | Không tính R01 lặp; R01 lấy từ V01; loại cell cũ ở các batch tiếp |

Thông tin trên nhằm giải thích số Saved Version khác số config thành công. Model/decoder trong retry không biến thành một idea mới, seed lặp không được tính thêm. Các lỗi đã được sửa và các batch cuối đều kết thúc thành công.
