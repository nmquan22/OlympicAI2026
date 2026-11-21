# ML: bảng, chuỗi thời gian, graph và quyết định

Mảng thứ tư bổ sung các bài không bắt đầu bằng fine-tune backbone. Nhãn ưu tiên trong [danh mục 2026](../docs/WORKSHOPS_2026.md) là đánh giá ôn tập, không phải xác suất ra đề.

| Khối | Dữ liệu → kết quả | Khi nên dùng |
|---|---|---|
| [Tabular pipeline](blocks/tabular.py) | cột số/category → lớp hoặc giá trị | Missingness, categorical, baseline linear/tree; preprocessing fit trên train |
| [Validation](blocks/validation.py) | group/time → indices; prediction → worst-group metric | Tách subject, machine, document, video; tránh random row split sai |
| [Time series](blocks/timeseries.py) | chuỗi → causal windows/statistics hoặc temporal logits | Forecasting, sensor classification, irregular/missingness features |
| [Graph](blocks/graphs.py) | adjacency + node features → node logits | Graph nhỏ, transductive với train labels; O(N²), không dùng cho graph lớn |
| [Comparison search](blocks/active_search.py) | embedding + phản hồi so sánh → query kế tiếp | Bài tương tác như Potato; soft update để chịu representation mismatch |
| [RL](blocks/rl.py) | transition/episode → loss hoặc Q-update | Ôn mở rộng **IOAI 2027**: Q-learning, DQN target, REINFORCE |

## Các baseline chạy offline

```bash
python -m ml.tabular --model linear
python -m ml.tabular --model trees
python -m ml.forecast
python -m ml.q_learning
```

Mặc định dùng dữ liệu tổng hợp để kiểm tra chạy code. Điểm in ra không phải điểm dataset thật.

CSV thật: train và valid có `id,target` cùng các cột feature. ID phải duy nhất; subject/scene khác nhau cần được chia từ trước. Cột category chưa thấy ở train được encode an toàn. Truyền `--categorical` không kèm giá trị để dùng chỉ số học.

```bash
python -m ml.tabular --train data/train.csv --valid data/valid.csv \
  --numeric age income --categorical region device --target target --model trees
python -m ml.tabular --train data/train.csv --valid data/valid.csv \
  --predict data/test.csv --numeric age income --categorical region --task regression
python -m ml.forecast --values data/series.npy --context 96 --horizon 24 --period 24
```

Tabular xuất model.joblib, metrics.json và predictions.csv. `--predict` nhận CSV không nhãn; metric vẫn chỉ tính trên valid. Khi so ensemble phải giữ nguyên row IDs và class order. Joblib checkpoint chỉ nên load từ nguồn mình kiểm soát.

Forecast nhận NPY `[time,channels]` đều thời gian, không NaN. Train target kết thúc trước cutoff 80%; validation origin bắt đầu ở cutoff. Gap loại các window có target băng qua split. File predictions.npz chứa origin để kiểm tra thời điểm. Với nhiều series, tạo window riêng từng ID, không nối cuối series này với đầu series khác.

## Hướng 2026 và bài mới từ IOAI

- [TSALM](../docs/workshops/tsalm.md): seasonal naive → ridge lag → context-aware forecast.
- [TS4H](../docs/workshops/ts4h.md): missingness, delta-time, subject split.
- [Potato và Robot Chasing](../docs/IOAI_2026.md): học từ feedback và behavior cloning. Robot Chasing là supervised snapshots; chỉ dùng RL khi đề có interaction/reward.
- [Syllabus mới](../docs/IOAI_2026.md#syllabus-công-bố-tháng-11): bổ sung RL được ghi rõ năm 2027.

Graph và RL là các khối mở rộng độ phủ. Việc có code trong repo không có nghĩa chúng thuộc mọi syllabus hoặc sẽ xuất hiện ở Olympic AI Việt Nam.
