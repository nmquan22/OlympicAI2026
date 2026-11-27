# Inverse Translation: tối ưu input rời rạc ở cùng mức điểm

Target text/token IDs và FSMT được đề cung cấp. Đây là tìm input cho frozen model, không phải train dịch thuật hay tự suy ra private target. R01/R04/R05/R06/R07 đều đạt100 public/private. R06 code11.97s so R01 16.29s trong một batch; warm-up khiến so thời gian một lần chỉ mang tính tham khảo. R02 top8 hụt hai target, R03 top16 hụt một target: bottleneck là recall của proposal từ probe một token, không chỉ số vòng search.

| Hạng | Paper | Phần áp dụng | Điều không tái lập |
|---|---|---|---|
| 1 | [HotFlip, Ebrahimi et al., ACL 2018](https://aclanthology.org/P18-2006.pdf) §3 | Xấp xỉ thay token bằng gradient theo input embedding, shortlist rồi exact evaluate | Task này là puzzle trên model cung cấp, không perturb văn bản người dùng. Dùng token-ID đúng vocab nguồn |
| 2 | [AutoPrompt, Shin et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.346.pdf) §2.2 | Gradient-guided candidate generation thay vì scan mọi token với model forward | Không dùng MLM/BERT khác; không tối ưu universal prompt cho dataset khác |
| 3 | [Universal Adversarial Triggers, Wallace et al., EMNLP 2019](https://aclanthology.org/D19-1221.pdf) §2 | Tối ưu token rời rạc, exact search sau first-order ranking | Không giả định cùng một trigger giải được24 target; mỗi row có input riêng |
| 4 | [Automatically Auditing Language Models via Discrete Optimization / ARCA, Jones et al., ICML 2023](https://proceedings.mlr.press/v202/jones23a/jones23a.pdf) §3 | Objective gần hành vi greedy và kiểm tra exact top-k, giảm sai lệch surrogate | Chưa cài averaged-gradient ARCA hay joint output optimization; target cố định nên không sửa target |
| 5 | [Hard Prompts Made Easy, Wen et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a00548031e4647b13042c97c922fadf1-Abstract-Conference.html) | Projected embedding optimization là phương án nếu local token swaps bị kẹt | Chưa cài PEZ; chỉ đọc abstract/source mô tả, PDF lớn không tải được qua retrieval. Không tuyên bố review đầy đủ thuật toán |

**R08:** dùng top8 grid search nhanh như R02. Với row chưa exact, backprop CE của target qua frozen FSMT tới embedding input tạm thời. Rank 23221 token bằng −grad·embedding; lấy16 token/position; exact-evaluate decision-margin của các phép thay. Chỉ chấp nhận nếu margin giảm. Max16 gradient steps. Nếu còn sai, fallback shortlist probe được ghi vào chi phí.

**R09:** khởi tạo từ top1 probe và dùng gradient substitutions ngay; tối đa24 bước. Dù ý tưởng thuộc họ coordinate optimization, code đánh giá toàn bộ single-position proposals rồi thay một vị trí tốt nhất mỗi bước, không phải sweep cập nhật mọi vị trí. Dùng cùng fallback top8.

Giữ canary4 examples, Transformers4.57.1, hook vị trí decoder đã kiểm, bỏ hook trước official evaluator. Không sửa model weights. Báo exact24/24, official score, gradient calls, model calls, candidate evaluations, walltime, peakVRAM. Chỉ kết luận tiết kiệm nếu vẫn100; nếu không thì ghi tradeoff, không gọi nhanh hơn là tốt hơn.
