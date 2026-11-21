# NLP · 30 bài luyện

Bắt đầu với UIT-VSFC để tự viết một pipeline phân loại. Chuyển sang PhoATIS khi cần một nhãn mỗi từ, rồi sang SQuAD khi đầu ra là hai vị trí trong văn bản. WikiText-2 dùng cùng tokenizer nhưng thay cả attention mask, target và metric.

## Học code theo thứ tự

| Khối | Đầu vào → đầu ra | Điều cần đọc |
|---|---|---|
| [JSONL, Dataset và collate](blocks/data.py) | hàng text/label → IDs có padding | Pad theo câu dài nhất trong batch |
| [BPE từ đầu](blocks/bpe.py) | từ → chuỗi symbol → token IDs | Đếm cặp kề theo tần suất từ, merge cặp tốt nhất, lưu thứ tự merge để encode lại |
| [CoNLL và CoNLL-U](blocks/sequence.py) | file tách câu bằng dòng trống → từ, BIO hoặc head–relation | Giữ ranh giới câu trước khi ghép batch |
| [TF-IDF + logistic](linear.py) | n-gram ký tự và từ → nhãn | Fit cả vocabulary và trọng số chỉ trên train |
| [Encoder và decoder-only](blocks/transformers.py) | IDs → class logits hoặc logits từ vựng tại từng vị trí | Encoder dùng masked mean. Decoder dùng triangular mask và nhãn dịch một bước |
| [Train và validation](train.py) | batch → loss → macro-F1 hoặc perplexity | Tổng NLL theo token hợp lệ trước khi tính perplexity |

Ví dụ, với `[BOS, học, máy, EOS]`, decoder nhận `[BOS, học, máy]` và dự đoán `[học, máy, EOS]`. Ở bước dự đoán *máy*, self-attention chỉ đọc tới *học*. Với encoder phân loại, tất cả token hợp lệ trong câu được nhìn thấy nhau. BPE trong repo hỗ trợ Unicode và có token UNK cho ký tự mới.

## Chạy offline

Chạy lệnh từ thư mục gốc repo. Khi bỏ các tham số dữ liệu, các lệnh sau dùng ví dụ nhỏ tích hợp.

```bash
python -m nlp.linear
python -m nlp.train --mode encoder --out runs/encoder
python -m nlp.train --mode lm --out runs/lm
```

Phân loại đọc hai file JSONL có `text` và `label` nguyên từ 0. LM chỉ cần `text`. Dùng `--train data/train.jsonl --valid data/valid.jsonl` với cùng các lệnh. Checkpoint và vocabulary của hai model ở hai thư mục riêng. Dữ liệu cho pretraining LM nên được chia theo tài liệu trước khi cắt thành chuỗi.

## Tái lập bài tiếng Việt

Lấy train/validation Parquet theo [manifest chính thức của UIT-VSFC](https://huggingface.co/api/datasets/uitnlp/vietnamese_students_feedback/parquet). Lưu cục bộ thành `data/train.parquet` và `data/valid.parquet`. Bộ chuyển đổi cần pandas và pyarrow trong requirements-extras.

```bash
python -m nlp.prepare_vsfc --train data/train.parquet --valid data/valid.parquet --limit 6000 --out data/vsfc
python -m nlp.linear --train data/vsfc/train.jsonl --valid data/vsfc/valid.jsonl
```

Trên CPU, 6.000 mẫu train stratified seed 7 và toàn bộ 1.583 mẫu validation chính thức cho **macro-F1 0,7448** và **accuracy 0,8901**. F1 lớp trung tính chỉ đạt **0,4082** trên 73 câu. Một ablation đáng làm là giữ nguyên split, tách char-only và word-only rồi xem recall trung tính thay đổi ra sao.

## 30 challenge theo thứ tự ưu tiên ôn

| Ưu tiên | Challenge | Metric cần theo dõi |
|---:|---|---|
| 1 | [UIT-VSFC · Cảm xúc phản hồi sinh viên](challenges/01-uitvsfc.md) | Macro-F1 ↑ |
| 2 | [PhoATIS · Intent và slot trong cùng câu](challenges/02-phoatis.md) | Intent accuracy, span F1 và exact match cả khung ↑ |
| 3 | [CoNLL-2003 · Nhận diện thực thể](challenges/03-conll.md) | Micro-F1 trên span khớp cả biên và loại ↑ |
| 4 | [SQuAD 1.1 · Trích đoạn trả lời](challenges/04-squad.md) | EM và token F1 ↑ |
| 5 | [PAWS · Phân biệt đảo vai nghĩa](challenges/05-paws.md) | Accuracy ↑ |
| 6 | [STS-B · Đo tương đồng theo thang liên tục](challenges/06-stsb.md) | Pearson r và Spearman ρ ↑ |
| 7 | [BANKING77 · Phân biệt intent gần nhau](challenges/07-banking.md) | Accuracy ↑ |
| 8 | [GoEmotions · Phân loại đa nhãn](challenges/08-emotion.md) | Macro-F1 ↑ |
| 9 | [UIT-ViQuAD · Đọc hiểu tiếng Việt](challenges/09-uitviqu.md) | EM và F1 theo evaluator của bộ dữ liệu ↑ |
| 10 | [Universal Dependencies · Phân tích phụ thuộc](challenges/10-ud.md) | LAS và UAS ↑ trên tokenization vàng của bài luyện |
| 11 | [SciERC · Quan hệ giữa thực thể khoa học](challenges/11-scierc.md) | Micro-F1 quan hệ dương ↑, dùng entity vàng |
| 12 | [SemEval 2014 · Tìm khía cạnh nhà hàng](challenges/12-absa.md) | Exact aspect-term span F1 ↑ |
| 13 | [XNLI · Suy diễn Anh–Việt](challenges/13-xnli.md) | Accuracy ↑ |
| 14 | [WiC · Một từ có cùng nghĩa?](challenges/14-wic.md) | Accuracy ↑ |
| 15 | [BoolQ · Đọc hiểu có/không](challenges/15-boolq.md) | Accuracy ↑ |
| 16 | [WinoGrande · Chọn từ điền đại từ](challenges/16-wsc.md) | Accuracy ↑ |
| 17 | [COPA · Chọn nguyên nhân hoặc hệ quả](challenges/17-copa.md) | Accuracy ↑ |
| 18 | [MultiRC · Chọn nhiều câu trả lời đúng](challenges/18-multirc.md) | F1a và exact match theo cả nhóm đáp án ↑ |
| 19 | [FEVER · Kiểm chứng bằng chứng](challenges/19-fever.md) | FEVER score ↑: đúng nhãn và đủ một evidence set cho claim kiểm chứng được |
| 20 | [SciFact · Truy hồi rationale khoa học](challenges/20-scifact.md) | Sentence-level selection+label F1 ↑ theo scorer SciFact |
| 21 | [TREC-COVID · Xếp hạng tài liệu](challenges/21-trec.md) | nDCG@10 ↑ |
| 22 | [MS MARCO · Passage reranking](challenges/22-msmarco.md) | MRR@10 ↑ |
| 23 | [TyDi QA GoldP · Span đa ngôn ngữ](challenges/23-tydi.md) | EM, token F1 ↑ theo ngôn ngữ trên GoldP |
| 24 | [HotpotQA · Đọc hai bằng chứng](challenges/24-hotpot.md) | Joint F1 ↑ |
| 25 | [XSum · Tóm tắt một câu](challenges/25-xsum.md) | ROUGE-1/2/L ↑ |
| 26 | [JFLEG · Sửa câu cho tự nhiên](challenges/26-jfleg.md) | GLEU ↑ theo script JFLEG, nhiều reference |
| 27 | [IWSLT Anh–Việt · Dịch câu TED](challenges/27-iwslt.md) | SacreBLEU ↑ |
| 28 | [WikiText-2 · Mô hình ngôn ngữ causal](challenges/28-wikitext.md) | Perplexity ↓ |
| 29 | [Spider · Sinh SQL từ schema](challenges/29-spider.md) | Execution accuracy ↑ |
| 30 | [CLINC150 · Từ chối yêu cầu ngoài miền](challenges/30-clinc.md) | In-scope accuracy và OOS recall ↑ |
