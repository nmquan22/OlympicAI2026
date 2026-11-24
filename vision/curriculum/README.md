# Vision nâng cao — học trực tiếp theo từng cell

16 module từ bộ tài liệu Vision, tích hợp vào repo. **Code tiếng Anh; giải thích tiếng Việt.** Mỗi notebook có dữ liệu tổng hợp nhỏ, forward, loss/update và chẩn đoán. Không tự tải dữ liệu/weights. Đây là code học cơ chế, chưa phải bản tái lập SOTA hay pipeline dữ liệu thật hoàn chỉnh.

Bắt đầu bằng [CV01: giải thích 11 cell](CV01_CELL_BY_CELL_VI.md), rồi [giải thích từng cell CV02–CV16](CELL_BY_CELL_VI.md). Đánh số cell code, không đếm markdown. Notebook cần Jupyter; các module và smoke test chỉ cần dependencies của repo.

```bash
# Chạy từ repo root
python -m vision.curriculum.smoke_test
python tools/check_curriculum_notebooks.py
```

| Bài | Chủ đề | Notebook | Module |
|---|---|---|---|
| CV01 | Object-centric learning / Slot Attention | [01](notebooks/cv01_slots.ipynb) | [cv01_slots.py](cv01_slots.py) |
| CV02 | Suy luận quy luật thị giác | [02](notebooks/cv02_reasoning.ipynb) | [cv02_reasoning.py](cv02_reasoning.py) |
| CV03 | Few-shot segmentation bằng ảnh hỗ trợ | [03](notebooks/cv03_visual_context.ipynb) | [cv03_visual_context.py](cv03_visual_context.py) |
| CV04 | Nhận dạng tổ hợp chưa thấy | [04](notebooks/cv04_composition.ipynb) | [cv04_composition.py](cv04_composition.py) |
| CV05 | Tín hiệu nhỏ trong video: rPPG | [05](notebooks/cv05_subtle.ipynb) | [cv05_subtle.py](cv05_subtle.py) |
| CV06 | Test-time adaptation | [06](notebooks/cv06_adapt.ipynb) | [cv06_adapt.py](cv06_adapt.py) |
| CV07 | Biến ảnh thành token rời rạc | [07](notebooks/cv07_tokenization.ipynb) | [cv07_tokenization.py](cv07_tokenization.py) |
| CV08 | Flow matching từ noise đến ảnh | [08](notebooks/cv08_flow.ipynb) | [cv08_flow.py](cv08_flow.py) |
| CV09 | Phát hiện ảnh tổng hợp | [09](notebooks/cv09_forensics.ipynb) | [cv09_forensics.py](cv09_forensics.py) |
| CV10 | Khôi phục nhiều loại suy giảm | [10](notebooks/cv10_restoration.ipynb) | [cv10_restoration.py](cv10_restoration.py) |
| CV11 | Ghép điểm giữa hai ảnh | [11](notebooks/cv11_matching.ipynb) | [cv11_matching.py](cv11_matching.py) |
| CV12 | Phân đoạn cả phần bị che | [12](notebooks/cv12_amodal.ipynb) | [cv12_amodal.py](cv12_amodal.py) |
| CV13 | Theo dõi điểm qua video | [13](notebooks/cv13_tracking.ipynb) | [cv13_tracking.py](cv13_tracking.py) |
| CV14 | Ghép ảnh và văn bản | [14](notebooks/cv14_open_vocab.ipynb) | [cv14_open_vocab.py](cv14_open_vocab.py) |
| CV15 | Dự đoán frame tương lai | [15](notebooks/cv15_world.ipynb) | [cv15_world.py](cv15_world.py) |
| CV16 | Học quên có đối chứng | [16](notebooks/cv16_unlearning.ipynb) | [cv16_unlearning.py](cv16_unlearning.py) |

Các notebook CV02–CV16 dùng năm cell code: (1) môi trường, (2) batch và model, (3) forward/baseline, (4) loss và update, (5) chẩn đoán và hướng nâng cấp. CV01 có 11 cell để đọc từng tensor.

## Dùng thư viện trước

[common.py](common.py) dùng torchvision ResNet18 và PyTorch MultiheadAttention/TransformerEncoder/GRU. [library_recipes.py](library_recipes.py) có torchvision Swin, DeepLab, Mask R-CNN, RAFT và transformers ViT/CLIP/GPT-2 tùy chọn. `transformers` không bắt buộc cho 16 notebook mặc định; các recipe đó chưa được kiểm tra bằng smoke test. Để dùng Hugging Face, cài transformers phù hợp môi trường và chuẩn bị local files trước khi thi.

`weights=None` là random initialization. Muốn tận dụng ImageNet, thay encoder bằng checkpoint được phép và dùng đúng preprocessing đi kèm. Swin stages xuất NHWC; cần permute về NCHW khi ghép CNN head. Hướng feature reconstruction của CV01 đang chờ teacher ResNet layer2 128 chiều; không thay ViT vào nguyên trạng.

## Giới hạn và cách học

- Tất cả số in trong notebook là fixture mechanics, không phải accuracy benchmark. Các biến thể chưa train không được so điểm để kết luận kiến trúc tốt hơn.
- Module có nhánh đã chạy và hướng nâng cấp mới chỉ là bài tập; xem [REFERENCE.md](REFERENCE.md). Các link nghiên cứu trong đó được giữ từ bộ tài liệu nguồn, chưa khảo sát lại ở lần tích hợp này.
- TTA cần source classifier đã học và protocol cho phép adaptation; default notebook chỉ mô phỏng cơ chế với random source.
- Model unlearning dùng uniform-forget objective minh họa, không chứng nhận xóa dữ liệu. Point tracking chưa có visibility head. World model mới có one-step teacher forcing.
- Chạy CPU trước; notebook mặc định không dùng AMP. Helper train_step chỉ bật BF16 trên CUDA có hỗ trợ BF16, để không giả định T4 hỗ trợ BF16.

Không thêm dataset/checkpoint hoặc log fixture cũ vào git. Kiểm thử hiện tại được ghi trong [VALIDATION.md](../../docs/VALIDATION.md).
