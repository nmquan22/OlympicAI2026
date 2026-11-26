# Cosmos Cosmic Run · Smoke test

Smoke test đã hoàn tất trên dữ liệu thật của Kaggle. Đây là luyện tập sau cuộc thi, không phải kết quả thi chính thức. Chín bài AICC khác mới được đưa vào hàng đợi.

## Kết quả quan sát

- [Notebook private, version 1](https://www.kaggle.com/code/shareaccc3a/aicc-cosmos-smoke-baseline-and-biome-validation?scriptVersionId=374506205), scriptVersionId `374506205`.
- [Trang submission](https://www.kaggle.com/competitions/cosmos-cosmic-run-aicc-round-11/submissions): **Complete (after deadline)**; public **0.72343**, private **0.71387**.
- Kaggle hiển thị runtime **32.4 giây** trên CPU; phần chương trình đo được **10.96 giây**. Chỉ một version đã chạy, không có draft session đang bật.
- Train: **4.773 dòng**, 420 races, 30 biomes. Test: **1.277 dòng**, 120 races, 39 biomes; cả 39 tên biome không có trong train. Hai file submission đều qua kiểm tra ID, số dòng và hoán vị hạng 1..n mỗi race.

| Mô hình | Holdout theo race | Holdout theo biome chưa thấy |
|---|---:|---:|
| CatBoost chỉ đặc trưng số | 0.585668 | 0.547287 |
| CatBoost + tên biome dạng categorical | 0.820048 | 0.556483 |
| Char TF-IDF × đặc trưng tàu + Ridge | **0.893597** | **0.821185** |

Điểm validation là trung bình Kendall tau từng race, rồi clip trung bình vào [0,1]; code dùng `scipy.stats.kendalltau`. Mỗi race có trọng số như nhau. Split race: 3.578 train / 1.195 validation, 105 races validation, có 30 biome dùng chung. Split biome: 3.365 train / 1.408 validation, 124 races validation, **0 biome dùng chung**. Cả hai giữ seed 42, holdout 25%.

## Flow và giả thuyết

1. Từ năm chỉ số tàu, thêm percentile rank và z-score trong cùng cuộc đua. Imputation chỉ fit trên phần train của split.
2. Baseline CatBoost: 600 iterations, depth 6, learning rate 0.05, CPU 2 threads; target là thứ hạng chuẩn hóa trong race.
3. Bản nâng cấp học TF-IDF `char_wb` n-gram 3–5, tối đa 1.500 đặc trưng từ tên biome trong train. Lấy tích mỗi đặc trưng số đã chuẩn hóa với vector tên biome, rồi Ridge alpha 10. Không dùng pretrained hoặc nhãn thủ công cho test.
4. Chọn mô hình bằng **validation biome**, fit lại toàn bộ train, dự đoán điểm cho từng tàu rồi xếp hạng trong race. Chỉ nộp mô hình đã chọn một lần để kiểm tra luồng chấm.

Tên biome dạng categorical giúp trên môi trường quen nhưng không mang quan hệ chữ sang tên mới. Tích TF-IDF với chỉ số tàu cho phép tác động của từng chỉ số thay đổi theo các mảnh tên môi trường. Đây là cách giải thích của thí nghiệm, chưa phải chứng minh nhân quả.

## Nguồn và khả năng tái lập

- [Đề và evaluator](https://www.kaggle.com/competitions/cosmos-cosmic-run-aicc-round-11/).
- [Notebook tham khảo của Anindyo](https://www.kaggle.com/code/anindyo237/notebookca40e6396b), Apache 2.0: dùng categorical biome, đặc trưng theo race và CatBoost. Trang nguồn hiển thị private 0.54761, race validation 0.82954; khác split/iterations với thí nghiệm này nên không xem là so sánh có kiểm soát.
- Lời giải Artix bị chặn bởi yêu cầu tài khoản thành viên; chưa đọc, không lấy số liệu từ đó. Đề công bố baseline 0.55 và reference 0.86; đó là số của ban tổ chức, không phải số mình tái lập.
- [Notebook kèm output](smoke.ipynb), [script](smoke.py), [stdout](run.log), [report gốc trích từ stdout](smoke_report.json), [bảng ablation](experiments.csv), [bằng chứng version/score](run_evidence.json).
- Notebook Kaggle giữ 21 output files, gồm prediction, split indices, metric từng race, report và model. Không đưa dataset gốc hoặc checkpoint vào repo.

Trên Kaggle, gắn competition data rồi chạy `smoke.ipynb` hoặc nội dung `smoke.py`. Các package cần dùng: numpy, pandas, scipy, scikit-learn, catboost, joblib. Checksum dữ liệu và phiên bản Python được ghi trong JSON. Dùng `python tools/aicc_record_run.py --report experiments/aicc/cosmos-cosmic-run/smoke_report.json --challenge cosmos-cosmic-run --notebook-url 'https://www.kaggle.com/code/shareaccc3a/aicc-cosmos-smoke-baseline-and-biome-validation?scriptVersionId=374506205' --version 1 --public-score 0.72343 --private-score 0.71387` để tái tạo nhật ký từ báo cáo.

## Giới hạn và điểm dừng

Chỉ có một holdout cho mỗi kiểu chia; chưa đo độ ổn định theo seed. Chênh validation biome 0.8212 với private 0.71387 cho thấy vẫn còn domain shift. Char n-gram không hiểu đầy đủ từ đồng nghĩa. Kết quả hiện tại đạt yêu cầu **smoke test toàn luồng**, chưa đạt reference 0.86.

Bước tiếp theo hợp lý: đo nhiều fold theo biome, so Ridge trên chênh lệch cặp tàu và target thời gian đã loại offset race; nếu đề cho phép, thử encoder ngữ nghĩa nhỏ. Chọn bằng validation, không tối ưu lặp lại theo private score. Chưa chạy các bước này và chưa đặt lịch chạy nền.

