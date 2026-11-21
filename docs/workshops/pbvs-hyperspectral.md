# PBVS: SR ảnh hyperspectral thực

**PBVS / CVPR 2026 · challenge · ưu tiên B · nguồn kiểm tra 2026-11-22.**

## Sự kiện đã xác minh

HISR là challenge mới, dùng cặp LR–HR thực và tỉ lệ x4. Nguồn xác nhận đây là lần đầu của workshop/challenge tại sự kiện này. [Nguồn chính thức](https://pbvs-workshop.github.io/challenge.html).

## Đề luyện do repo đề xuất

Cube [bands,h,w] → cube [bands,4h,4w].

Metric để luyện: **PSNR và spectral angle cho bài luyện; đọc scorer track.** Đây là lựa chọn cho bài rút gọn; cần scorer và rules của đúng track để báo điểm challenge chính thức.

| Bước | Cách làm và mục đích |
|---|---|
| Baseline | Bicubic từng band rồi residual CNN nhận nhiều kênh. |
| Nâng cấp vừa sức | Nén phổ bằng PCA fit trên train; chia sẻ spatial encoder và dự đoán residual phổ. |
| Hướng nghiên cứu để đọc tiếp | Đọc mạng kết hợp spatial–spectral và degradation estimation. |

Phép thử quyết định bước tiếp theo: **So spectral angle dù PSNR tăng; kiểm tra thứ tự và scale từng band.**

## Chuẩn bị và khối code

Link Codabench 12418 trên trang PBVS. Thay channels của ResidualRestorer; loader cube cần viết theo dữ liệu thực.

- [vision/blocks/restoration.py](../../vision/blocks/restoration.py)
- [ml/blocks/validation.py](../../ml/blocks/validation.py)

Các khối có thể cần adapter theo schema thật. Chúng không phải lời giải chính thức của workshop và không tự tải dataset/weights. Chạy fixture chỉ kiểm tra đường chạy; điểm fixture không đại diện benchmark.

Áp dụng [ngân sách 300 phút](../PRACTICE_PLAN.md) và ghi cả score, thời gian, seed, split và số mẫu. [Về danh mục 2026](../WORKSHOPS_2026.md).
