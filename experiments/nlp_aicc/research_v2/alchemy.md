# Alchemy: thiếu kiến thức nghĩa và bài toán matching

## Bài khó ở đâu?

Chỉ có 150 quy tắc, 70 cặp test và 70 candidate result, mỗi result dùng một lần. Train fold có 120 quy tắc; val 30 tên kết quả tách rời. Cross-encoder scratch không có tri thức rằng hai nguyên tố kết hợp thành vật gì. Tăng depth/hard negatives không tạo ra tri thức thiếu. R09 shared roles + additive composition đạt val 5/30, Kaggle public .0285/private .1142. Val nhỏ, kết quả này chưa ổn định.

R06 Sinkhorn cũ chuẩn hóa minibatch 8 query ×16 candidate; không thể vừa có tổng mỗi hàng 1 và mỗi cột 1 trên ma trận chữ nhật ấy. Đây là mismatch objective với giả định bijection, không phải thử nghiệm sạch cho matching vuông. Hungarian chỉ tối ưu ma trận score đã có; score không mang nghĩa thì assignment vẫn sai.

## 5 paper ưu tiên

| Hạng | Paper / nguồn gốc | Phần phù hợp | Giới hạn / loại bỏ | Lượt |
|---|---|---|---|---|
| 1 | [Learning Latent Permutations with Gumbel-Sinkhorn Networks, Mena et al., ICLR 2018](https://arxiv.org/html/1802.08665v1) §2–3 | Relaxation trên ma trận vuông doubly stochastic | Không gọi normalization 8×16 là permutation; không thêm Gumbel noise khi không cần latent sampling | R10 balanced deterministic Sinkhorn |
| 2 | [Deep Sets, Zaheer et al., NeurIPS 2017](https://arxiv.org/html/1703.06114v3) §2–3 | Hàm ghép bất biến khi đổi item1/item2; shared item encoder | Symmetry chỉ bớt sample complexity, không tạo kiến thức vật lý | Giữ composer đối xứng R09 trong R10 |
| 3 | [An Embarrassingly Simple Approach to Zero-shot Learning, Romera-Paredes & Torr, ICML 2015](https://proceedings.mlr.press/v37/romera-paredes15.html) | Bilinear compatibility giữa feature query và thuộc tính candidate, ridge hai phía | Bài gốc có semantic attributes. BPE/character ở đây chỉ là lexical attributes, không tương đương semantic zero-shot | R11 kiểm tra giới hạn của lexical compatibility |
| 4 | [SuperGlue, Sarlin et al., CVPR 2020](https://arxiv.org/html/1911.11763v2) §3.2–3.3 | Học score cùng ràng buộc assignment thay vì sửa ở inference | Không dùng pretrained CV, geometry hoặc dustbin vì bài yêu cầu bijection đầy đủ | R10 row/column CE + Sinkhorn NLL |
| 5 | [Translating Embeddings for Modeling Multi-relational Data, Bordes et al., NeurIPS 2013](https://proceedings.neurips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf) §2 | Shared entity space, low-capacity additive compatibility | item2 là entity, không phải relation như TransE; unseen disconnected names không thể dựa riêng entity ID | Giữ shared lexical roles; không gọi a+b là TransE reproduction |

## Thiết kế đóng băng trước scoring

**R10:** giữ nguyên composer R09. Minibatch chọn 16 result khác nhau và một rule cho mỗi result, tạo đúng score16×16. Row CE + column CE với inverse permutation, cộng .2×Sinkhorn NLL sau 20 log-normalization. Canary kiểm tổng hàng/cột≈1; assert target duy nhất. Cuối cùng vẫn Hungarian. 240 steps/180 giây, BPE train fold, cùng holdout và seed.

**R11:** fit BPE + TF-IDF character2–4 trên tên trong training fold, SVD≤64 chỉ fit training, query=[a+b, |a−b|, a*b, bias]. Giải bilinear compatibility bằng ridge hai phía alpha=beta=1 trên T4. Candidate chưa thấy vẫn có lexical attributes. Đây là phép thử xem chia sẻ hình thái có đủ không, không giả định tên gần chữ là gần nghĩa. Không grid search bằng 30 val hoặc leaderboard.

Cả hai in coverage: bao nhiêu tên val/test từng xuất hiện trong training ở bất kỳ vai trò nào, bao nhiêu từng làm input, số UNK. Không thêm cạnh heldout/label test, không tải BERT hay tokenizer pretrained. Chưa có điểm GPU vòng 2 tại thời điểm chốt thiết kế.
