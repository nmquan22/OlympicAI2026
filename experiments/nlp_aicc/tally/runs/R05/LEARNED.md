R05 teacher KL: val spanF1 .370370 (TP5/FP7/FN10), code490.202s,92.551MiB. 1848 weak batches có mean countMAE trước/sau .27017166/.15912460 và mean teacher KL.27616628. Đây là trung bình train, không phải holdoutF1. Weak holdout MAE.25225/vector29.25% đều kémR03; ORG0/5, PERSON chỉ3/7. Sáu bước dual đã kéo moment nhưng không định vị được span. Không giữ nâng cấp này, không gọi exact projection hoặc thành công vì residual giảm.

Bằng chứng: số được đọc từ output Kaggle thật; bản lưu này là tóm tắt các trường đo, không giả lập notebook đã execute. Xem version trong evidence.json để đọc toàn bộ log.

Kaggle đã Complete(after deadline): public 0.25020, private 0.25609; private chỉ báo cáo.
