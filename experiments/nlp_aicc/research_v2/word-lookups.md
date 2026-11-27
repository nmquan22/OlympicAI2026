# Word Lookups: từ biên BPE đến từ ngôn ngữ

Train không có nhãn segmentation. R07 kết hợp hai chiều, PMI, normalization, DP và punctuation đạt Boundary F1 .9097/.9113. Không có gold để phân loại lỗi test từng câu; chỉ có thể đặt giả thuyết từ cơ chế: biên BPE là đơn vị nén, không bắt buộc là biên từ; một bigram PMI không đủ nhận biết một cụm ba/bốn ký tự dùng độc lập. Không gọi những câu đoán sai chưa biết là error analysis có nhãn.

| Hạng | Paper | Điểm cần học | Quyết định / điều kiện |
|---|---|---|---|
| 1 | [Accessor Variety Criteria for Chinese Word Extraction, Feng et al., CL 2004](https://aclanthology.org/J04-1004.pdf) §4 | Cụm có nhiều predecessor/successor khác nhau có dấu hiệu dùng độc lập; AV=min(left,right) | R08 dùng AV làm bonus cho span2–4. Khai thác train corpus, không dùng từ điển ngoài |
| 2 | [Unsupervised Neural Word Segmentation for Chinese via Segmental Language Modeling, Sun & Deng, EMNLP 2018](https://aclanthology.org/D18-1531.pdf) §2.4 | Training cộng xác suất mọi segmentation bằng DP, decoding tìm segmentation tốt nhất | R08 vẫn heuristic span score; không gọi DP decoder của nó là SLM đã marginalize latent words |
| 3 | [Learning to Discover, Ground and Use Words with Segmental Neural Language Models, Kawakami et al., ACL 2019](https://aclanthology.org/P19-1645.pdf) §3–4 | Lexical memory + character decoder; regularizer ngăn lexicon học nguyên câu | Chưa cài memory generative. Giới hạn span và tần suất trong R08 là hypothesis riêng, không thay thế objective bài gốc |
| 4 | [Contextual Dependencies in Unsupervised Word Segmentation, Goldwater et al., COLING-ACL 2006](https://aclanthology.org/P06-1085.pdf) §3–4 | Word-to-word context và inference procedure ảnh hưởng undersegmentation; unigram objective có thể thích ghép collocation | Không dùng Gibbs/HDP trong lượt này; dùng để kiểm tra nguy cơ AV làm ghép quá tay |
| 5 | [BPE-Dropout, Provilkov et al., ACL 2020](https://aclanthology.org/2020.acl-main.170.pdf) §3 | Bỏ ngẫu nhiên một số merge khi train giúp model không lệ thuộc một cách chia subword | R09 dropout .1 khi train, standard BPE khi val/infer. BLEU của NMT không thể đem làm dự báo F1 segmentation |

**R08:** giữ R07; đếm substring2–4 trên 90% training fold, frequency≥5, AV≥2; sentence boundary được tính theo số lần xuất hiện đầu/cuối câu. Bonus span=min(.5,.2 log(1+AV)) cộng vào DP. Ngưỡng cố định trước scoring, không scan leaderboard. Nhãn BMES vẫn offset-safe, punctuation giữ biên bắt buộc.

**R09:** giữ R08, BPE-dropout .1 và rematerialize tokenization mỗi epoch; tắt dropout khi validation/inference. Tokenizers có RNG merge riêng nên seed42 không cam kết tái lập bitwise qua phiên bản thư viện. Ghi rõ điều này. Không nhìn nhãn test, không chọn threshold từ leaderboard.

Mỗi config tự train LM từ đầu, tối đa8 epoch/240 giây rồi decode9272 câu. Dự kiến khoảng4–5 phút/config theo R07, thêm startup mỗi batch. Chấm chính bằng Kaggle Boundary F1; NLL không phải segmentation F1.
