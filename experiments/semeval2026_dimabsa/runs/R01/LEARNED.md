# R01 · BiLSTM attention: tốt hơn Transformer hiện tại, chưa thắng Ridge

**Dev RMSE_VA 1.50198** với scorer BTC; [Kaggle Version 1](https://www.kaggle.com/code/quannguyen0501/dimabsa-r01-scratch-bilstm-aspect-attention?scriptVersionId=374890167). [Report](report.json), [log epoch](epochs.jsonl), [scorer](official_scorer.log).

Model: BPE fit train sạch → embedding random 192 → BiLSTM 2 tầng, hidden 96 mỗi chiều → mean/max context + aspect pooling + attention theo aspect → MLP → VA. 1.169.476 tham số. MSE, AdamW LR 0.002, dropout 0.2, warmup 10% + cosine, seed 42. Best epoch 12, early-stop sau epoch 22. Model/preprocessing 44.67 giây; Saved Version 83.2 giây, chưa tính queue.

R01 và R02 dùng **cùng tokenizer hash**, cùng 2.336 cặp train sạch/340 cặp dev; encoder depth, parameter count và LR khác nhau nên đây là so hai recipe trong ngân sách tương tự, không phải ablation chỉ thay đúng một toán tử.

| Model | Dev RMSE_VA ↓ |
|---|---:|
| Ridge với cửa sổ aspect | 1.45154 |
| BiLSTM attention R01 | 1.50198 |
| Transformer pre-norm R02 | 1.62459 |
| Trung bình cố định 50/50 Ridge + R01 | 1.39469 |

Phép gộp cuối tính từ prediction cache, không train thêm và không tìm trọng số tốt nhất. Chưa chấm test; cùng dev đã được dùng chọn epoch nên đây là diagnostic để định hướng, không phải ước lượng generalization độc lập.

Lỗi còn lớn ở V < 5 (RMSE_VA 2.61687) và câu > 30 từ (3.32131). Loss train tiếp tục giảm khi dev đã chững; thêm epoch không phải cải tiến có căn cứ. Chọn BiLSTM làm điểm xuất phát neural cho Subtask 2–3, nơi gold có span opinion và liên kết, thay vì cấp thêm lượt chỉ tăng chiều rộng/độ sâu.

Bài học phòng thi: so với control rẻ, đọc nhóm lỗi và kiểm tra khả năng bổ sung prediction. Không suy ra LSTM luôn tốt hơn GRU hoặc Transformer từ phép thử này; hai nhánh BiGRU đã bỏ **trước khi chạy** theo yêu cầu của Quân.
