# 5 notebook học ý tưởng và tái lập

Các notebook chứa code đầy đủ, không tải source từ GitHub lúc chạy. Chúng chưa execute trong bản export này; điểm bên dưới được dẫn từ các Saved Version đã chạy thật. Mỗi notebook kèm bảng5 paper, giả định áp dụng, cấu hình đối chứng và hạn chế.

| Notebook | Preset mở sẵn | Điểm Kaggle public/private đã đo | Câu hỏi để học |
|---|---|---|---|
| [moment-match_study.ipynb](moment-match_study.ipynb) | R12 Soft-DTW | .6456/.6253 | Vì sao giữ token interaction và thứ tự event? Tăng3 câu val có đủ tin không? |
| [lost-interpreter_study.ipynb](lost-interpreter_study.ipynb) | R08 BPE BiGRU | MAE14.1608/16.7920 | Vì sao recurrence hợp dữ liệu này? Tại sao typed inputs/curriculum chưa thắng? |
| [word-lookups_study.ipynb](word-lookups_study.ipynb) | R08 accessor spans | Boundary F1 .9150/.9171 | Vì sao decoder tăng F1 khi NLL không đổi? |
| [inverse-translation_study.ipynb](inverse-translation_study.ipynb) | R06 coordinate | 100/100 | Cắt shortlist có thể mất recall; giảm calls ở cùng điểm như thế nào? |
| [alchemy_study.ipynb](alchemy_study.ipynb) | R10 square matching | .0285/.1142 | Bijection và symmetry giải quyết phần nào? 0 UNK có nghĩa là biết nghĩa không? |

Alchemy là ví dụ để học giới hạn, chưa phải lời giải tốt. Moment R12 tốt nhất val nhưng chưa vượt private R04. Không so điểm giữa các bài khác metric.

## Cách chạy

1. Import notebook vào Kaggle, attach data của competition tương ứng; tên slug có trong source `find_data`.
2. Chọn GPU T4. `start_run` xác minh loại GPU trước khi chạy. Internet cần cho cài Transformers4.57.1 ở Inverse nếu image chưa có; model FSMT vẫn chỉ đọc checkpoint được đề cung cấp.
3. Đổi `RUN_ID` nếu muốn thử một config khác có trong bảng `*_CONFIGS`, rồi Run All từ đầu. Không chạy hai config trên cùng model/optimizer.
4. Đọc `run_report.json`, diagnostics và `submission.csv` ở `/kaggle/working/<task>/Rxx/`. Kiểm schema/IDs thành công mới nộp. Không chọn preset theo private score.

T4 code đã đo: Lost R08≈146s, Word R08≈231s, Moment R12≈87s, Alchemy R10≈14s, Inverse R06≈12s. Thêm startup/package/evaluator; queue không nằm trong các con số này. BatchWord2 lượt mất8m23s thực tế. Khi queue, giãn mốc kiểm2→4→8→15 phút thay vì polling liên tục.

`MANIFEST.json` ghi hash source và commit. `python tools/nlp_make_study_notebooks.py` tái tạo5 file và AST-check mọi code cell; đây là kiểm cấu trúc/source, không phải huấn luyện lại trên T4.
