# PBVS: thermal SR có RGB hướng dẫn

**PBVS / CVPR 2026 · workshop · ưu tiên A · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

TISR có track thermal-only x8 và track RGB-guided x8/x16. Đã xác minh phiên bản năm 2026; không suy ra đây là workshop mới thành lập. [Nguồn chính thức](https://pbvs-workshop.github.io/challenge.html).

## Đề luyện do repo đề xuất

LR thermal + RGB đã căn chỉnh → HR thermal một kênh.

Metric để luyện: **PSNR/MAE cho bài luyện; đối chiếu scorer chính thức của track.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Bicubic thermal; đo riêng trường hợp bỏ nhánh RGB. |
| Nâng cấp vừa sức | Gated RGB fusion và residual thermal; augment hình học đồng bộ hai nhánh. |
| Hướng nghiên cứu để đọc tiếp | Đọc cross-attention và alignment để giảm copy nhầm texture RGB. |

Phép thử quyết định bước tiếp theo: **Dịch RGB 2–4 pixel ở dev; nếu score sụp, ưu tiên alignment và gate.**

## Chuẩn bị và khối code

Dữ liệu TISR theo link Track 2 trên trang nguồn; giữ scene tách biệt. GuidedSR là khối from-scratch, chưa tái lập leaderboard.

- [vision/blocks/restoration.py](../../vision/blocks/restoration.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
