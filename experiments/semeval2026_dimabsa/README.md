# DimABSA 2026 · Luyện scratch từ hồi quy đến trích xuất

Phạm vi Quân chọn ngày 27/11/2026: **Track A, tiếng Anh–restaurant; bắt đầu Subtask 1 rồi mở rộng Subtask 2–3.** Không dùng pretrained weights, embeddings, tokenizer có sẵn, teacher hoặc LLM tạo dữ liệu. Có thể dùng PyTorch và thư viện xây model với random initialization. Tối đa **hai phiên Kaggle Saved Version hoạt động cùng lúc**, tính cả phiên người dùng đã mở.

## Trạng thái đã xác minh

- GitHub đã kết nối, có quyền ghi repo.
- Đã tải train/dev/test từ commit BTC `bdc93be1224106ae7d3eb95739c02a76ed4ae8a1`; [manifest](DATA_MANIFEST.json) có SHA256 trên bản chuẩn hóa newline LF.
- **Đã chạy ba control CPU trên dữ liệu thật**, đã đối chiếu bằng scorer BTC. Xem [report](runs/cpu_controls/report.json), [log scorer](runs/cpu_controls/C02_official_scorer.log).
- **Chưa upload Kaggle, chưa khởi chạy GPU.** Đang chờ người dùng hoàn tất xác minh đăng nhập Google. Code neural mới được kiểm tra cú pháp, chưa kiểm thử forward/backward hay đo GPU.
- Chưa chấm test và chưa nộp Codabench. Điểm dưới đây là **official dev split, chấm cục bộ bằng scorer BTC**; không phải leaderboard.

## Control CPU để biết GPU cần vượt mốc nào

Giữ cùng 2.336 cặp text–aspect train sau làm sạch và 340 cặp dev. Alpha Ridge cố định 3.0; không quét siêu tham số. Mỗi prediction làm tròn 2 chữ số trước chấm.

| Run | Model | RMSE_VA chính thức ↓ | RMSE V ↓ | RMSE A ↓ |
|---|---|---:|---:|---:|
| C00 | Trung bình VA của train | 1.83654 | 1.63785 | 0.83086 |
| C01 | Word/char TF-IDF toàn câu + aspect, Ridge | 1.45520 | 1.22043 | 0.79257 |
| C02 | TF-IDF cửa sổ quanh aspect + aspect, Ridge | 1.45154 | 1.21576 | 0.79303 |

**Bài học hiện có:** đặc trưng từ/ký tự đã mang tín hiệu sentiment đáng kể. Cửa sổ quanh aspect chỉ đổi rất ít điểm ở phép thử này; chưa đủ bằng chứng kết luận nó tốt hơn. Model neural cần đánh giá cả V và A, cùng mức chi phí; không mặc định Transformer scratch sẽ thắng Ridge hoặc BiGRU trên tập nhỏ.

## Đọc solution rồi chuyển idea sang điều kiện scratch

Các thứ hạng/điểm trong bảng này là **kết quả tác giả dưới điều kiện pretrained**, tách khỏi kết quả tự chạy ở trên. Chuyển idea là giả thuyết cần ablation, không phải tái lập điểm của tác giả.

| Nguồn gốc | Điều đã xác minh | Idea có thể thử khi scratch |
|---|---|---|
| [LogSigma](https://aclanthology.org/2026.semeval-1.165/) và [PDF](https://aclanthology.org/2026.semeval-1.165.pdf) | Top 1 English restaurant/laptop ST1; encoder pretrained + hai head VA + learned homoscedastic uncertainty + ensemble seed | Hai tham số log-variance cân bằng loss V/A; so với MSE bằng cùng encoder/seed. Ensemble chỉ sau khi hiểu từng model. |
| PICT, abstract chính thức trong [SemEval 2026 proceedings](https://aclanthology.org/events/semeval-2026/) | Top 3 restaurant ST1 theo overview; weighted-layer pooling, attention theo aspect, residual regression, R-Drop/SWA trên RoBERTa-Large | Attention dùng aspect làm query; pooling các tầng là thí nghiệm riêng sau. Không giữ backbone pretrained. |
| [YangSteam](https://aclanthology.org/2026.semeval-1.199/) | Aspect-marker encoding và dual regression heads trên encoder pretrained | Đánh dấu/biểu diễn aspect tường minh, để cùng một câu có thể cho VA khác nhau theo mục tiêu. Chưa đọc toàn bộ implementation của nhóm. |
| Takoyaki, abstract trong [proceedings](https://aclanthology.org/events/semeval-2026/) và [overview](https://aclanthology.org/2026.semeval-1.452/) | Top 1 English ST2/ST3; BM25 demonstration retrieval + agreement ensemble của LLM | Khi sang ST2/ST3, học ý tưởng voting/ghép tuple và khử trùng. Phần LLM ICL không áp dụng trong điều kiện scratch. |

## Hai lượt GPU đầu và bước quyết định

| Run | Kiến trúc | So sánh muốn trả lời | Trạng thái |
|---|---|---|---|
| R01 | BPE train-only → embedding random → 2-layer BiGRU → mean/max text + aspect pool → head VA | Encoder có học được composition hơn n-gram không? | prepared, chưa chạy |
| R02 | R01 + attention context dùng aspect làm query | Tập trung bằng query aspect có giúp câu nhiều aspect không? | prepared, chưa chạy |
| R03 | Thay BiGRU bằng Transformer nhỏ pre-norm random init | Self-attention có lợi trên tập nhỏ trong cùng budget không? | planned, chưa chạy |
| R04 | Encoder được chọn + learned log-variance trên hai loss | Điều chỉnh gradient V/A có tốt hơn MSE bằng nhau không? | planned, chưa chạy |

Mỗi lượt dùng cùng split/seed, fit lại tokenizer bằng train sạch và khởi tạo mới model/optimizer. R01/R02 dùng MSE, output `1 + 8*sigmoid(logit)` và chọn epoch theo RMSE_VA dev. Giới hạn mặc định 40 epoch, patience 6, khoảng 20 phút/lượt; có thể dừng sau epoch hiện tại nên không phải hard timeout. R03/R04 chỉ quyết định sau khi đọc R01/R02 và nhóm lỗi; không mặc định chạy hết mọi tổ hợp.

Khi đã chọn xong kiến trúc trên dev, mới chạy inference test, chấm bằng gold test cục bộ và ghi riêng kết quả. Mở rộng ST2: encoder → span aspect/opinion → liên kết cặp → head VA; ST3 thêm head category. Phải có scorer cF1 và phân tích phần mất điểm do span, liên kết, category, VA trước khi thêm kiến trúc phức tạp. Các phần ST2/ST3 **chưa implement**.

## Ba chi tiết phòng thi cần nhớ

1. **Đọc schema trước model:** train có 2.284 câu và 3.659 quadruplet, gồm 880 nhãn aspect NULL. Bài explicit-aspect bỏ NULL; nhiều opinion của cùng aspect được gộp VA bằng trung bình. Đây là policy chuyển nhãn được ghi rõ, không giả định train đã là schema Subtask 1.
2. **Kiểm tra trùng split:** có một câu trùng giữa train và dev ST1 sau lowercase/chuẩn hóa khoảng trắng. Bỏ câu đó khỏi train cho mọi control và model, còn 2.336 cặp text–aspect; giữ nguyên toàn bộ dev.
3. **Dùng đúng metric:** scorer tính `sqrt(mean(error_V**2 + error_A**2))`, lớn hơn RMSE gộp tọa độ một hệ số √2. Lưu cả hai để tránh so nhầm paper; các PCC với dự đoán hằng không xác định nên ghi null.

## Tái lập

```bash
python experiments/semeval2026_dimabsa/prepare_data.py --download --out /tmp/dimabsa_data --manifest /tmp/dimabsa_manifest.json
python experiments/semeval2026_dimabsa/cpu_baselines.py --data /tmp/dimabsa_data --out /tmp/dimabsa_cpu
python experiments/semeval2026_dimabsa/train_scratch.py --data /tmp/dimabsa_data --out /tmp/dimabsa_R01 --run R01
python experiments/semeval2026_dimabsa/train_scratch.py --data /tmp/dimabsa_data --out /tmp/dimabsa_R02 --run R02 --pooling attention
```

Neural cần `torch`, `tokenizers`, `numpy`, `scipy`, `scikit-learn`. Chưa khóa phiên bản Kaggle vì chưa có runtime GPU thực tế; mỗi report sẽ ghi torch version, GPU, tokenizer hash, số parameter, thời gian, epoch, config, split và score. Scorer BTC tải từ `evaluation_script/metrics_subtask_1_2_3.py` cùng source commit; không thay công thức chấm. Không commit dataset thô, API token hoặc checkpoint lớn vào repo.
