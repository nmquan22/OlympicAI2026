# Audio: sound, speech và temporal grounding

Mảng thứ năm luyện những bài cần nghe nhưng không nhất thiết cần ASR. [DCASE 2026](../docs/WORKSHOPS_2026.md#audio) có đủ bảy task trong danh mục, từ classification đến spatial audio; độ khó và mức hỗ trợ code được ghi riêng.

| Khối | Input → output | Giới hạn cần hiểu |
|---|---|---|
| [Frontend](blocks/features.py) | WAV → mono 16kHz → log-mel; SpecAugment | Mono bỏ spatial cues; không dùng loader này nguyên trạng cho spatial audio |
| [AudioCNN](blocks/models.py) | `[B,mel,T]` + lengths → class logits | Mask sau mỗi layer để padding không đổi pooling |
| [CTC greedy](blocks/models.py) | `[B,T,vocab]` → token sequences | Collapse repeats trước khi xóa blank; đây là decoder, chưa có ASR trainer |
| [Edit distance](blocks/models.py) | word/character lists → lỗi | Corpus WER = tổng lỗi / tổng từ reference, không lấy trung bình WER từng câu |
| [Temporal decoder](blocks/temporal.py) | clip scores + boundaries → interval | Center cosine hoặc fit length penalty, tránh luôn chọn đoạn dài nhất |

## Chạy classification

```bash
python -m audio.classify
python -m audio.classify --train data/audio/train.csv --valid data/audio/valid.csv \
  --epochs 20 --device cuda --out runs/sound
```

CSV có `id,path,label`, label integer `0..C-1`; đường WAV tương đối với thư mục chứa CSV. Train và valid phải khác source recording/speaker/machine, không chỉ khác crop. Dùng clip ngắn khi luyện; CLI cache toàn bộ log-mel trong RAM. Với dataset lớn, lưu feature ra đĩa hoặc dùng Dataset đọc theo batch.

Mặc định tạo các tín hiệu tone tổng hợp với noise và chiều dài khác nhau. Chỉ dùng để kiểm tra forward/backward/export, không phải benchmark âm thanh tự nhiên. Log-mel hiện là HTK-style power dB; nếu thay bằng pretrained encoder, dùng frontend mà model đó yêu cầu.

Output: `model.pt` và `predictions.csv`. Metadata ghi sample rate/classes. Với multilabel hoặc hierarchy, thay head/loss/evaluator theo task; không dùng clip macro-F1 thay metric challenge.

## Thứ tự luyện gợi ý

1. [DCASE 1](../docs/workshops/dcase1.md): classification có duration/missingness khác nhau.
2. [DCASE 2](../docs/workshops/dcase2.md): normal-only kNN trên audio descriptors.
3. [DCASE 6](../docs/workshops/dcase6.md): temporal grounding với cached features do organizer cấp.
4. [Find the Order](../docs/IOAI_2026.md): audio/ASR features → directed pair scores → permutation.
5. [DCASE 5](../docs/workshops/dcase5.md) và [DCASE 7](../docs/workshops/dcase7.md): audio QA có control silence; continual learning theo đúng no-replay rules.

Task 3/4 được giữ ở mức C vì cần multi-channel frontend, spatial scoring và pipeline chuyên biệt. Repo có các khối khởi đầu, chưa có lời giải hoàn chỉnh của hai task này.
