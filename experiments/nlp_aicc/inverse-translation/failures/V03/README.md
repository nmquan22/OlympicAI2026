# V03: search thành công, wrapper còn cell cũ

Canary 4/4 khớp. Transformers 4.57.1, tokenizers 0.22.2: search top32 đạt official 100, exact24/24, code16.7748s, probe3.7728s,9624 candidate evaluations,182 model calls,359.12MiB peak.

Sau BATCH_RESULT hoàn tất, cell cũ còn sót tiếp tục cài transformers4.44.2/tokenizers0.19.1 và thất bại trên Python3.13. Saved Version 3 (scriptVersionId374702151) bị đánh Failed. Chưa tính vào số lượt hoàn tất; chạy lại R01 trong batch sạch cùng R02–R07.

Sửa: select all, xóa nội dung cũ, paste source mới, so text chuẩn hóa bằng source mong đợi trước Save Version. Không bỏ canary, không gỡ assertion model.
