# TSALM: forecasting có context

**TSALM / ICLR 2026 · workshop · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

TSALM lần đầu ở ICLR, bàn time series với large models và context. Nguồn xác nhận đây là lần đầu của workshop/challenge tại sự kiện này. [Nguồn chính thức](https://tsalm-workshop.github.io/).

## Đề luyện do repo đề xuất

Chuỗi quá khứ + context đã biết tại dự báo → horizon tương lai.

Metric để luyện: **MAE và seasonal-naive relative error; rolling-origin eval.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Seasonal naive và ridge trên lag windows. |
| Nâng cấp vừa sức | Thêm calendar/exogenous features; kiểm tra feature có biết ở thời điểm dự đoán. |
| Hướng nghiên cứu để đọc tiếp | Đọc time-series foundation model nếu có weights; không bỏ statistical controls. |

Phép thử quyết định bước tiếp theo: **So ridge/temporal model với seasonal naive cùng horizon và timestamps.**

## Chuẩn bị và khối code

Chỉ regular sequence không NaN trong CLI; irregular data cần adapter và mask. Không cho train target vượt cutoff.

- [ml/forecast.py](../../ml/forecast.py)
- [ml/blocks/timeseries.py](../../ml/blocks/timeseries.py)
- [ml/blocks/validation.py](../../ml/blocks/validation.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
