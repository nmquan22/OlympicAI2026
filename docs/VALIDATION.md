# Phạm vi kiểm tra đợt mở rộng 2026

## Bổ sung Vision ngày 24/11/2026

[Curriculum Vision](../vision/curriculum/README.md) được chạy trên Python **3.12**, PyTorch **2.5.1+cpu**, torchvision **0.20.1+cpu**:

- **16 notebook, 86 cell code** chạy tuần tự từ đầu; bao gồm forward, loss/backward/update, baseline controls và chẩn đoán. Không cần Jupyter/transformers cho script kiểm tra.
- Smoke test của **16 module** chạy các biến thể baseline/nâng cấp trên tensor tổng hợp, gồm RGB/feature reconstruction, reasoning, prototypes/cross-attention, TENT, VQ, flow và các hướng còn lại. Random feature teacher là control kiểm tra code, không phải pretrained DINOSAUR.
- **27 test** hiện có và **12 fixture entry points** của repo đều pass; kiểm tra syntax, Markdown local links, diff whitespace và render registry nhất quán đều pass.
- Chưa train/eval dataset thật cho 16 chủ đề; chưa kiểm tra GPU, optional transformers loaders hoặc tái lập kết quả paper/SOTA. Một bước gradient trên fixture không chứng minh model tốt hơn baseline.

```bash
python -m vision.curriculum.smoke_test
python tools/check_curriculum_notebooks.py
```

## Đợt mở rộng trước

Kiểm tra trên CPU, Python 3.12, PyTorch **2.15.0+cpu**, torchvision **0.29.2+cpu**, scikit-learn **1.8.0**. Không có GPU trong phiên chạy này. Dữ liệu fixture tổng hợp chỉ kiểm tra code; chưa đo leaderboard hoặc hiệu năng T4/H100 MIG.

| Kiểm tra | Nội dung |
|---|---|
| 22 behavioral tests | Exhaustive order search nhỏ, order→rank không tự nghịch đảo, Unicode offsets, time/group leakage, CTC blank/repeat, WAV resampling, timestamp IoU, graph permutation, RL terminal/truncation, selective perturbation, padding invariance, missing-modality NaNs, tile/full equivalence, gradient flow, calibration và parameter budget |
| 5 file integration tests | Nhánh đọc CSV, WAV manifest, ảnh LR/HR, JSONL labels tách riêng và cached NPY; inference không nhãn, thứ tự ID và rank convention |
| 12 entry points | 7 recipe mới + 5 pipeline gốc; mỗi lệnh chạy process riêng, ghi output vào temporary directory |
| Registry và tài liệu | 39 IDs duy nhất, tất cả block paths tồn tại, syntax Python và local Markdown links hợp lệ |
| Render nhất quán | JSON registry render lại ra đúng workshop index/cards đã commit |

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v
python tools/check_repo.py
python tools/smoke_recipes.py
python tools/render_catalog.py
git diff --exit-code -- docs/WORKSHOPS_2026.md docs/workshops
```

12 entry points gồm `ml.tabular`, `ml.forecast`, `ml.q_learning`, `nlp.order`, `nlp.boundary`, `audio.classify`, `vision.restore`, `nlp.train` (encoder), `vision.detect`, `vision.segment`, `vision_ml.retrieve`, `vision_ml.anomaly`. Các chế độ/dataset chưa liệt kê không được coi là đã chạy trong đợt này.

Workflow [checks.yml](../.github/workflows/checks.yml) tái chạy kiểm tra khi push/PR. Kết quả GitHub Actions cần xem trực tiếp trên GitHub; việc có workflow không tự chứng minh CI đã xanh.

Chưa tải dữ liệu/weights của các workshop mới, chưa tái lập SOTA và chưa thực hiện đầy đủ official-format adapters. Dùng bảng [BLOCKS.md](BLOCKS.md) để phân biệt chương trình có thể chạy ngay với khối cần ghép. Các nguồn 2026 có link và ngày kiểm tra trong registry, không có script âm thầm scrape/download trong lúc thi.
