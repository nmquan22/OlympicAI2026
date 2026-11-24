# Diễn giải các notebook theo từng cell

Đánh số **cell code**, không đếm markdown. Mỗi ví dụ tự tạo fixture, chạy offline bằng CPU và không tải weights/dataset. Tất cả điểm trên fixture chỉ kiểm tra cơ chế.


## CV02 — Suy luận quy luật thị giác

Tám ô ngữ cảnh và A đáp án. Mỗi đáp án được ghép thành một chuỗi 9 ô, chấm điểm riêng; dùng CE trên A điểm. Split theo template/quy luật để phát hiện shortcut.


### Cell 1 — Môi trường và seed

Dùng CPU để kiểm tra trước. Tìm repo root từ thư mục notebook, đặt seed và import PyTorch. Không tải pretrained weights.

```python
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'vision/curriculum').is_dir())
sys.path.insert(0, str(ROOT))
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device('cpu')  # Change to cuda after the CPU example works.
print('Device:', device)
```


### Cell 2 — Tạo batch và model

Tám ô ngữ cảnh và A đáp án. Mỗi đáp án được ghép thành một chuỗi 9 ô, chấm điểm riêng; dùng CE trên A điểm. Split theo template/quy luật để phát hiện shortcut. Dữ liệu trong cell là fixture tổng hợp để bạn đọc được shape; thay bằng Dataset thật trước khi so metric.

```python
from vision.curriculum.cv02_reasoning import Reasoner, loss_fn
context = torch.rand(2, 8, 3, 32, 32, device=device)
choices = torch.rand(2, 4, 3, 32, 32, device=device)
labels = torch.tensor([0, 1], device=device)
model = Reasoner(mode='relation').to(device)
batch = (context, choices, labels)
```


### Cell 3 — Forward và baseline

Đọc shape/output trước khi train. Các số in ra chỉ minh họa phép tính và luồng dự đoán.

```python
model.eval()
with torch.no_grad():
    scores = model(context, choices)
print('Scores B x A:', scores.shape)
print('Candidate indices:', scores.argmax(-1).tolist())
```


### Cell 4 — Một bước cập nhật

Tính đúng loss của nhiệm vụ, backward và cập nhật tham số. Một bước không phải quy trình huấn luyện hoàn chỉnh. TENT tự thực hiện update bên trong adapt.

```python
model.train()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model, *batch)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
print('One-step training loss:', loss.detach().item())
```


### Cell 5 — Chẩn đoán và hướng nâng cấp

MLP trên 9 embeddings → all-pairs relation → Transformer. Phép đổi thứ tự đáp án phải đổi đúng thứ tự score; accuracy trên fixture ngẫu nhiên không có ý nghĩa.

```python
model.eval()
with torch.no_grad():
    permutation = torch.tensor([2, 0, 3, 1], device=device)
    original = model(context, choices)
    permuted = model(context, choices[:, permutation])
    error = (permuted - original[:, permutation]).abs().max()
print('Candidate permutation error:', float(error))
print('Baseline -> relation -> transformer: change mode; retrain on the same split.')
```


## CV03 — Few-shot segmentation bằng ảnh hỗ trợ

Support có mask, query chưa có mask lúc suy luận. Học theo episode; lớp novel không trùng lớp train. Prototype là masked average feature, còn cross-attention giữ nhiều support tokens.


### Cell 1 — Môi trường và seed

Dùng CPU để kiểm tra trước. Tìm repo root từ thư mục notebook, đặt seed và import PyTorch. Không tải pretrained weights.

```python
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'vision/curriculum').is_dir())
sys.path.insert(0, str(ROOT))
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device('cpu')  # Change to cuda after the CPU example works.
print('Device:', device)
```


### Cell 2 — Tạo batch và model

Support có mask, query chưa có mask lúc suy luận. Học theo episode; lớp novel không trùng lớp train. Prototype là masked average feature, còn cross-attention giữ nhiều support tokens. Dữ liệu trong cell là fixture tổng hợp để bạn đọc được shape; thay bằng Dataset thật trước khi so metric.

```python
from vision.curriculum.cv03_visual_context import VisualContext, loss_fn
support = torch.rand(2, 1, 3, 32, 32, device=device)
support_mask = torch.zeros(2, 1, 1, 32, 32, device=device)
support_mask[..., 8:24, 8:24] = 1
query = support[:, 0].clone()
target = support_mask[:, 0].clone()
model = VisualContext(mode='prototype').to(device)
batch = (support, support_mask, query, target)
```


### Cell 3 — Forward và baseline

Đọc shape/output trước khi train. Các số in ra chỉ minh họa phép tính và luồng dự đoán.

```python
model.eval()
with torch.no_grad():
    logits = model(support, support_mask, query)
print('Query mask logits:', logits.shape)
print('Foreground probability mean:', logits.sigmoid().mean().item())
```


### Cell 4 — Một bước cập nhật

Tính đúng loss của nhiệm vụ, backward và cập nhật tham số. Một bước không phải quy trình huấn luyện hoàn chỉnh. TENT tự thực hiện update bên trong adapt.

```python
model.train()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model, *batch)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
print('One-step training loss:', loss.detach().item())
```


### Cell 5 — Chẩn đoán và hướng nâng cấp

Prototype FG/BG → query-to-support attention. Chấm mIoU theo protocol của bộ dữ liệu; IoU fixture chỉ minh họa công thức.

```python
model.eval()
with torch.no_grad():
    prediction = model(support, support_mask, query).sigmoid() > 0.5
    truth = target.bool()
    intersection = (prediction & truth).sum()
    union = (prediction | truth).sum()
print('Fixture binary IoU:', (intersection / union.clamp_min(1)).item())
print('Upgrade: VisualContext(mode="cross"); train and evaluate novel-class episodes.')
```


## CV04 — Nhận dạng tổ hợp chưa thấy

Tách thuộc tính và đối tượng: thấy red car và blue chair trong train rồi đánh giá red chair. Candidate pairs ở test phải bao gồm các tổ hợp cần dự đoán.


### Cell 1 — Môi trường và seed

Dùng CPU để kiểm tra trước. Tìm repo root từ thư mục notebook, đặt seed và import PyTorch. Không tải pretrained weights.

```python
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'vision/curriculum').is_dir())
sys.path.insert(0, str(ROOT))
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device('cpu')  # Change to cuda after the CPU example works.
print('Device:', device)
```


### Cell 2 — Tạo batch và model

Tách thuộc tính và đối tượng: thấy red car và blue chair trong train rồi đánh giá red chair. Candidate pairs ở test phải bao gồm các tổ hợp cần dự đoán. Dữ liệu trong cell là fixture tổng hợp để bạn đọc được shape; thay bằng Dataset thật trước khi so metric.

```python
from vision.curriculum.cv04_composition import Composition, loss_fn
images = torch.rand(2, 3, 32, 32, device=device)
pairs = torch.tensor([[0, 0], [1, 1], [0, 1]], device=device)
labels = torch.tensor([0, 1], device=device)
model = Composition(n_attr=2, n_obj=2, mode='factorized').to(device)
batch = (images, pairs, labels)
```


### Cell 3 — Forward và baseline

Đọc shape/output trước khi train. Các số in ra chỉ minh họa phép tính và luồng dự đoán.

```python
model.eval()
with torch.no_grad():
    scores, attribute_scores, object_scores = model(images, pairs)
print('All pair scores:', scores.shape)
print('Predicted attribute/object pairs:', pairs[scores.argmax(-1)].tolist())
```


### Cell 4 — Một bước cập nhật

Tính đúng loss của nhiệm vụ, backward và cập nhật tham số. Một bước không phải quy trình huấn luyện hoàn chỉnh. TENT tự thực hiện update bên trong adapt.

```python
model.train()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model, *batch)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
print('One-step training loss:', loss.detach().item())
```


### Cell 5 — Chẩn đoán và hướng nâng cấp

Flat classifier là control hạn chế → attribute/object factorization → learned pair embeddings. Fixture chưa tách seen/unseen; không dùng nó để báo generalized CZSL accuracy.

```python
model.eval()
with torch.no_grad():
    scores, _, _ = model(images, pairs)
print('Score for unseen candidate (0, 1):', scores[:, 2].tolist())
print('Upgrade: mode="joint" adds learned compositional embeddings.')
print('Real evaluation: seen accuracy, unseen accuracy, harmonic mean, calibration AUC.')
```


## CV05 — Tín hiệu nhỏ trong video: rPPG

Video ROI da [B,T,3,H,W] và waveform đồng bộ [B,T]. Dùng split theo người. Phân giải tần số phụ thuộc thời lượng cửa sổ, không chỉ FPS.


### Cell 1 — Môi trường và seed

Dùng CPU để kiểm tra trước. Tìm repo root từ thư mục notebook, đặt seed và import PyTorch. Không tải pretrained weights.

```python
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'vision/curriculum').is_dir())
sys.path.insert(0, str(ROOT))
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device('cpu')  # Change to cuda after the CPU example works.
print('Device:', device)
```


### Cell 2 — Tạo batch và model

Video ROI da [B,T,3,H,W] và waveform đồng bộ [B,T]. Dùng split theo người. Phân giải tần số phụ thuộc thời lượng cửa sổ, không chỉ FPS. Dữ liệu trong cell là fixture tổng hợp để bạn đọc được shape; thay bằng Dataset thật trước khi so metric.

```python
from vision.curriculum.cv05_subtle import PulseNet, rgb_trace, chrom, heart_rate, loss_fn
fps = 30
steps = 300
time = torch.arange(steps, device=device) / fps
wave = torch.sin(2 * torch.pi * 1.2 * time)
video = torch.full((2, steps, 3, 8, 8), 0.5, device=device)
video[:, :, 1] += 0.02 * wave[None, :, None, None]
target = wave[None].expand(2, -1)
model = PulseNet().to(device)
batch = (video, target)
```


### Cell 3 — Forward và baseline

Đọc shape/output trước khi train. Các số in ra chỉ minh họa phép tính và luồng dự đoán.

```python
green = rgb_trace(video)[..., 1]
print('Green baseline BPM:', heart_rate(green, fps).tolist())
print('Simplified CHROM BPM:', heart_rate(chrom(video), fps).tolist())
print('Target BPM:', 72)
```


### Cell 4 — Một bước cập nhật

Tính đúng loss của nhiệm vụ, backward và cập nhật tham số. Một bước không phải quy trình huấn luyện hoàn chỉnh. TENT tự thực hiện update bên trong adapt.

```python
model.train()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model, *batch)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
print('One-step training loss:', loss.detach().item())
```


### Cell 5 — Chẩn đoán và hướng nâng cấp

Green trace → CHROM đơn giản → 3D CNN với correlation và frequency loss. Ví dụ có 10 giây video; không dùng 4 frame để suy nhịp tim.

```python
model.eval()
with torch.no_grad():
    predicted_wave = model(video)
print('Model waveform:', predicted_wave.shape)
print('Untrained/one-step model BPM:', heart_rate(predicted_wave, fps).tolist())
print('Report subject-disjoint HR MAE only after real training.')
```


## CV06 — Test-time adaptation

Source classifier đã train gặp ảnh bị corruption. TENT cố định trọng số convolution, chỉ cập nhật affine của BatchNorm bằng entropy. Chỉ áp dụng khi luật cho phép học từ input test.


### Cell 1 — Môi trường và seed

Dùng CPU để kiểm tra trước. Tìm repo root từ thư mục notebook, đặt seed và import PyTorch. Không tải pretrained weights.

```python
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'vision/curriculum').is_dir())
sys.path.insert(0, str(ROOT))
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device('cpu')  # Change to cuda after the CPU example works.
print('Device:', device)
```


### Cell 2 — Tạo batch và model

Source classifier đã train gặp ảnh bị corruption. TENT cố định trọng số convolution, chỉ cập nhật affine của BatchNorm bằng entropy. Chỉ áp dụng khi luật cho phép học từ input test. Dữ liệu trong cell là fixture tổng hợp để bạn đọc được shape; thay bằng Dataset thật trước khi so metric.

```python
from copy import deepcopy
from torchvision.models import resnet18
from vision.curriculum.cv06_adapt import configure_tent, adapt, entropy
images = torch.rand(2, 3, 32, 32, device=device)
source_model = resnet18(weights=None, num_classes=2).to(device).eval()
model = deepcopy(source_model)
optimizer = configure_tent(model)
```


### Cell 3 — Forward và baseline

Đọc shape/output trước khi train. Các số in ra chỉ minh họa phép tính và luồng dự đoán.

```python
with torch.no_grad():
    source_logits = source_model(images)
print('Source entropy:', entropy(source_logits).mean().item())
print('Trainable BN parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
```


### Cell 4 — Một bước cập nhật

Tính đúng loss của nhiệm vụ, backward và cập nhật tham số. Một bước không phải quy trình huấn luyện hoàn chỉnh. TENT tự thực hiện update bên trong adapt.

```python
# adapt computes entropy, backward and optimizer.step internally.
before = next(p for p in model.parameters() if p.requires_grad).detach().clone()
logits_before_update = adapt(model, optimizer, images)
after = next(p for p in model.parameters() if p.requires_grad).detach()
print('Adapted logits:', logits_before_update.shape)
print('BN affine change:', (after - before).norm().item())
```


### Cell 5 — Chẩn đoán và hướng nâng cấp

Frozen source → BN statistics → TENT entropy minimization → entropy filtering (chưa phải toàn bộ EATA). Phải có source checkpoint đã học để đánh giá adaptation.

```python
model = deepcopy(source_model)
optimizer = configure_tent(model)
print('Reset to immutable source before the next domain.')
print('No labels are consumed by adapt; random source weights are only a mechanics demo.')
```


## CV07 — Biến ảnh thành token rời rạc

Encoder giảm kích thước 4 lần mỗi chiều. Mỗi vector được thay bằng codebook entry gần nhất. Straight-through chuyển gradient reconstruction về encoder; hai VQ terms học codebook và commitment.


### Cell 1 — Môi trường và seed

Dùng CPU để kiểm tra trước. Tìm repo root từ thư mục notebook, đặt seed và import PyTorch. Không tải pretrained weights.

```python
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'vision/curriculum').is_dir())
sys.path.insert(0, str(ROOT))
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device('cpu')  # Change to cuda after the CPU example works.
print('Device:', device)
```


### Cell 2 — Tạo batch và model

Encoder giảm kích thước 4 lần mỗi chiều. Mỗi vector được thay bằng codebook entry gần nhất. Straight-through chuyển gradient reconstruction về encoder; hai VQ terms học codebook và commitment. Dữ liệu trong cell là fixture tổng hợp để bạn đọc được shape; thay bằng Dataset thật trước khi so metric.

```python
from vision.curriculum.cv07_tokenization import VQVAE, loss_fn
images = torch.rand(2, 3, 32, 32, device=device)
model = VQVAE(d=32, k=64).to(device)
batch = (images,)
```


### Cell 3 — Forward và baseline

Đọc shape/output trước khi train. Các số in ra chỉ minh họa phép tính và luồng dự đoán.

```python
with torch.no_grad():
    reconstruction, indices, vq_loss = model(images)
print('Reconstruction:', reconstruction.shape)
print('Raster token indices:', indices.shape)
print('VQ loss:', vq_loss.item())
```


### Cell 4 — Một bước cập nhật

Tính đúng loss của nhiệm vụ, backward và cập nhật tham số. Một bước không phải quy trình huấn luyện hoàn chỉnh. TENT tự thực hiện update bên trong adapt.

```python
model.train()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model, *batch)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
print('One-step training loss:', loss.detach().item())
```


### Cell 5 — Chẩn đoán và hướng nâng cấp

Continuous AE → VQ-VAE → causal LM trên raster tokens → residual quantization/perceptual loss. Theo dõi dead codes, không chỉ PSNR.

```python
model.eval()
with torch.no_grad():
    reconstruction, indices, _ = model(images)
    counts = torch.bincount(indices.flatten(), minlength=model.code.num_embeddings).float()
    probabilities = counts / counts.sum()
    perplexity = (-(probabilities * probabilities.clamp_min(1e-9).log()).sum()).exp()
print('Used codebook entries:', (counts > 0).sum().item())
print('Fixture codebook perplexity:', perplexity.item())
print('Next: train a causal token LM on indices; optional transformers recipe in library_recipes.py.')
```


## CV08 — Flow matching từ noise đến ảnh

Nối noise z và ảnh x bằng x_t=(1-t)z+t*x. Target là vận tốc x-z; sampler bắt đầu t=0 rồi cộng vận tốc theo dt dương tới t=1.


### Cell 1 — Môi trường và seed

Dùng CPU để kiểm tra trước. Tìm repo root từ thư mục notebook, đặt seed và import PyTorch. Không tải pretrained weights.

```python
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'vision/curriculum').is_dir())
sys.path.insert(0, str(ROOT))
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device('cpu')  # Change to cuda after the CPU example works.
print('Device:', device)
```


### Cell 2 — Tạo batch và model

Nối noise z và ảnh x bằng x_t=(1-t)z+t*x. Target là vận tốc x-z; sampler bắt đầu t=0 rồi cộng vận tốc theo dt dương tới t=1. Dữ liệu trong cell là fixture tổng hợp để bạn đọc được shape; thay bằng Dataset thật trước khi so metric.

```python
from vision.curriculum.cv08_flow import Velocity, loss_fn, sample
images = torch.rand(2, 3, 32, 32, device=device) * 2 - 1
model = Velocity(d=32).to(device)
batch = (images,)
```


### Cell 3 — Forward và baseline

Đọc shape/output trước khi train. Các số in ra chỉ minh họa phép tính và luồng dự đoán.

```python
noise = torch.randn_like(images)
times = torch.tensor([0.25, 0.75], device=device)
interpolated = (1 - times[:, None, None, None]) * noise + times[:, None, None, None] * images
velocity = model(interpolated, times)
print('Velocity field:', velocity.shape)
print('Target velocity:', (images - noise).shape)
```


### Cell 4 — Một bước cập nhật

Tính đúng loss của nhiệm vụ, backward và cập nhật tham số. Một bước không phải quy trình huấn luyện hoàn chỉnh. TENT tự thực hiện update bên trong adapt.

```python
model.train()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model, *batch)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
print('One-step training loss:', loss.detach().item())
```


### Cell 5 — Chẩn đoán và hướng nâng cấp

CNN vận tốc → time-conditioned U-Net → conditional flow restoration/solver tốt hơn. Khôi phục ảnh bằng flow cần nhánh điều kiện; ví dụ này chưa có.

```python
model.eval()
with torch.no_grad():
    generated = sample(model, torch.randn_like(images), steps=8)
print('Euler samples:', generated.shape)
print('Finite samples:', torch.isfinite(generated).all().item())
print('One gradient step cannot generate meaningful images; this checks the ODE path only.')
```


## CV09 — Phát hiện ảnh tổng hợp

So RGB classifier với RGB + FFT log-amplitude. Bắt buộc tách generator/source image và giữ JPEG/resize cân bằng hai lớp để tránh shortcut.


### Cell 1 — Môi trường và seed

Dùng CPU để kiểm tra trước. Tìm repo root từ thư mục notebook, đặt seed và import PyTorch. Không tải pretrained weights.

```python
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'vision/curriculum').is_dir())
sys.path.insert(0, str(ROOT))
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device('cpu')  # Change to cuda after the CPU example works.
print('Device:', device)
```


### Cell 2 — Tạo batch và model

So RGB classifier với RGB + FFT log-amplitude. Bắt buộc tách generator/source image và giữ JPEG/resize cân bằng hai lớp để tránh shortcut. Dữ liệu trong cell là fixture tổng hợp để bạn đọc được shape; thay bằng Dataset thật trước khi so metric.

```python
from vision.curriculum.cv09_forensics import Forensics, loss_fn
images = torch.rand(2, 3, 32, 32, device=device)
labels = torch.tensor([0, 1], device=device)
model = Forensics(mode='fusion').to(device)
batch = (images, labels)
```


### Cell 3 — Forward và baseline

Đọc shape/output trước khi train. Các số in ra chỉ minh họa phép tính và luồng dự đoán.

```python
model.eval()
with torch.no_grad():
    scores = model(images)
    spectrum = torch.fft.fftshift(torch.fft.fft2(images).abs(), dim=(-2, -1)).log1p()
print('Binary logits:', scores.shape)
print('Log-amplitude spectrum:', spectrum.shape)
```


### Cell 4 — Một bước cập nhật

Tính đúng loss của nhiệm vụ, backward và cập nhật tham số. Một bước không phải quy trình huấn luyện hoàn chỉnh. TENT tự thực hiện update bên trong adapt.

```python
model.train()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model, *batch)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
print('One-step training loss:', loss.detach().item())
```


### Cell 5 — Chẩn đoán và hướng nâng cấp

RGB → RGB+FFT → patch/multi-scale features. Độ chính xác cao trong cùng generator chưa chứng minh khả năng bắt generator mới.

```python
model.eval()
with torch.no_grad():
    fake_probability = model(images).softmax(-1)[:, 1]
print('Fixture fake probabilities:', fake_probability.tolist())
print('Compare mode="rgb" and mode="fusion" on the same held-out generator split.')
print('Report AUROC/balanced accuracy on real data; two random images cannot measure generalization.')
```


## CV10 — Khôi phục nhiều loại suy giảm

Ảnh sạch y, ảnh suy giảm x có cùng tọa độ. Mô hình dự đoán residual, cộng vào x. Conditioning dự đoán scale/bias từ feature toàn ảnh, tương tự FiLM.


### Cell 1 — Môi trường và seed

Dùng CPU để kiểm tra trước. Tìm repo root từ thư mục notebook, đặt seed và import PyTorch. Không tải pretrained weights.

```python
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'vision/curriculum').is_dir())
sys.path.insert(0, str(ROOT))
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device('cpu')  # Change to cuda after the CPU example works.
print('Device:', device)
```


### Cell 2 — Tạo batch và model

Ảnh sạch y, ảnh suy giảm x có cùng tọa độ. Mô hình dự đoán residual, cộng vào x. Conditioning dự đoán scale/bias từ feature toàn ảnh, tương tự FiLM. Dữ liệu trong cell là fixture tổng hợp để bạn đọc được shape; thay bằng Dataset thật trước khi so metric.

```python
from vision.curriculum.cv10_restoration import Restoration, loss_fn
clean_images = torch.rand(2, 3, 32, 32, device=device)
degraded = (clean_images + 0.1 * torch.randn_like(clean_images)).clamp(0, 1)
model = Restoration(d=32, condition=True).to(device)
batch = (degraded, clean_images)
```


### Cell 3 — Forward và baseline

Đọc shape/output trước khi train. Các số in ra chỉ minh họa phép tính và luồng dự đoán.

```python
model.eval()
with torch.no_grad():
    restored = model(degraded)
print('Input/output:', degraded.shape, restored.shape)
print('Identity baseline MSE:', F.mse_loss(degraded, clean_images).item())
```


### Cell 4 — Một bước cập nhật

Tính đúng loss của nhiệm vụ, backward và cập nhật tham số. Một bước không phải quy trình huấn luyện hoàn chỉnh. TENT tự thực hiện update bên trong adapt.

```python
model.train()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model, *batch)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
print('One-step training loss:', loss.detach().item())
```


### Cell 5 — Chẩn đoán và hướng nâng cấp

Identity → residual CNN → feature conditioning → Restormer/PromptIR là hướng đọc thêm, chưa tái lập trong module.

```python
model.eval()
with torch.no_grad():
    restored = model(degraded).clamp(0, 1)
    baseline_mse = (degraded - clean_images).square().mean((1, 2, 3))
    restored_mse = (restored - clean_images).square().mean((1, 2, 3))
print('Fixture identity PSNR:', (-10 * baseline_mse.clamp_min(1e-12).log10()).mean().item())
print('Fixture model PSNR:', (-10 * restored_mse.clamp_min(1e-12).log10()).mean().item())
print('Compare condition=False/True, then try a trained multi-scale restoration architecture.')
```


## CV11 — Ghép điểm giữa hai ảnh

Mỗi source token phân loại vị trí token đích. Correlation [B,Nsrc,Ndst], target [B,Nsrc], -100 đánh dấu source không có nhãn. Hai ảnh ở đây giống nhau nên dùng ánh xạ identity.


### Cell 1 — Môi trường và seed

Dùng CPU để kiểm tra trước. Tìm repo root từ thư mục notebook, đặt seed và import PyTorch. Không tải pretrained weights.

```python
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'vision/curriculum').is_dir())
sys.path.insert(0, str(ROOT))
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device('cpu')  # Change to cuda after the CPU example works.
print('Device:', device)
```


### Cell 2 — Tạo batch và model

Mỗi source token phân loại vị trí token đích. Correlation [B,Nsrc,Ndst], target [B,Nsrc], -100 đánh dấu source không có nhãn. Hai ảnh ở đây giống nhau nên dùng ánh xạ identity. Dữ liệu trong cell là fixture tổng hợp để bạn đọc được shape; thay bằng Dataset thật trước khi so metric.

```python
from vision.curriculum.cv11_matching import Matcher, loss_fn, soft_coordinates
first = torch.rand(2, 3, 32, 32, device=device)
second = first.clone()
model = Matcher(context=False).to(device)
with torch.no_grad():
    feature_height, feature_width = model.enc(first).shape[-2:]
labels = torch.arange(feature_height * feature_width, device=device).repeat(2, 1)
batch = (first, second, labels)
```


### Cell 3 — Forward và baseline

Đọc shape/output trước khi train. Các số in ra chỉ minh họa phép tính và luồng dự đoán.

```python
model.eval()
with torch.no_grad():
    correlations = model(first, second)
    coordinates = soft_coordinates(correlations, feature_height, feature_width)
print('Correlation:', correlations.shape)
print('Expected destination coordinates in feature pixels:', coordinates.shape)
```


### Cell 4 — Một bước cập nhật

Tính đúng loss của nhiệm vụ, backward và cập nhật tham số. Một bước không phải quy trình huấn luyện hoàn chỉnh. TENT tự thực hiện update bên trong adapt.

```python
model.train()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model, *batch)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
print('One-step training loss:', loss.detach().item())
```


### Cell 5 — Chẩn đoán và hướng nâng cấp

Cosine features → supervised correlation → cross-attention → coarse-to-fine. Soft coordinate đang dùng pixel trên lưới feature, chưa phải pixel ảnh gốc.

```python
model.eval()
with torch.no_grad():
    correlations = model(first, second)
    identity_accuracy = (correlations.argmax(-1) == labels).float().mean()
print('Identical-image feature match accuracy:', identity_accuracy.item())
print('Upgrade: context=True; then coarse-to-fine refinement. Real metrics: PCK or homography error.')
```


## CV12 — Phân đoạn cả phần bị che

Input RGB và visible mask, target full mask. Mô hình chỉ học hidden region; công thức union đảm bảo xác suất full mask chứa visible mask nhị phân.


### Cell 1 — Môi trường và seed

Dùng CPU để kiểm tra trước. Tìm repo root từ thư mục notebook, đặt seed và import PyTorch. Không tải pretrained weights.

```python
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'vision/curriculum').is_dir())
sys.path.insert(0, str(ROOT))
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device('cpu')  # Change to cuda after the CPU example works.
print('Device:', device)
```


### Cell 2 — Tạo batch và model

Input RGB và visible mask, target full mask. Mô hình chỉ học hidden region; công thức union đảm bảo xác suất full mask chứa visible mask nhị phân. Dữ liệu trong cell là fixture tổng hợp để bạn đọc được shape; thay bằng Dataset thật trước khi so metric.

```python
from vision.curriculum.cv12_amodal import Amodal, loss_fn, predict
images = torch.rand(2, 3, 32, 32, device=device)
full = torch.zeros(2, 1, 32, 32, device=device)
full[..., 8:24, 8:24] = 1
visible = full.clone()
visible[..., 16:, :] = 0
model = Amodal(d=16).to(device)
batch = (images, visible, full)
```


### Cell 3 — Forward và baseline

Đọc shape/output trước khi train. Các số in ra chỉ minh họa phép tính và luồng dự đoán.

```python
with torch.no_grad():
    hidden_logits = model(images, visible)
    full_probability = predict(model, images, visible)
print('Hidden logits:', hidden_logits.shape)
print('Visible area / full target area:', visible.sum().item(), full.sum().item())
```


### Cell 4 — Một bước cập nhật

Tính đúng loss của nhiệm vụ, backward và cập nhật tham số. Một bước không phải quy trình huấn luyện hoàn chỉnh. TENT tự thực hiện update bên trong adapt.

```python
model.train()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model, *batch)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
print('One-step training loss:', loss.detach().item())
```


### Cell 5 — Chẩn đoán và hướng nâng cấp

Visible mask → hidden-region CNN → shape prior/copy-paste occlusion. Nếu đề không cấp visible mask thì cần detector/segmenter phía trước.

```python
model.eval()
with torch.no_grad():
    full_probability = predict(model, images, visible)
print('Contains the visible region:', (full_probability >= visible).all().item())
print('Baseline hidden-region recall is zero when predicting visible only.')
print('Evaluate full IoU and invisible-region IoU separately after training.')
```


## CV13 — Theo dõi điểm qua video

Lấy query feature tại tọa độ frame đầu rồi correlate với các frame. Tọa độ chuẩn hóa [-1,1], thứ tự x,y, align_corners=False; pixel center i được đổi bằng 2*(i+0.5)/size-1.


### Cell 1 — Môi trường và seed

Dùng CPU để kiểm tra trước. Tìm repo root từ thư mục notebook, đặt seed và import PyTorch. Không tải pretrained weights.

```python
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'vision/curriculum').is_dir())
sys.path.insert(0, str(ROOT))
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device('cpu')  # Change to cuda after the CPU example works.
print('Device:', device)
```


### Cell 2 — Tạo batch và model

Lấy query feature tại tọa độ frame đầu rồi correlate với các frame. Tọa độ chuẩn hóa [-1,1], thứ tự x,y, align_corners=False; pixel center i được đổi bằng 2*(i+0.5)/size-1. Dữ liệu trong cell là fixture tổng hợp để bạn đọc được shape; thay bằng Dataset thật trước khi so metric.

```python
from vision.curriculum.cv13_tracking import Tracker, loss_fn
frame = torch.rand(2, 1, 3, 32, 32, device=device)
video = frame.repeat(1, 4, 1, 1, 1)
points = torch.tensor([[[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]], device=device).repeat(2, 1, 1)
trajectory = points[:, None].expand(-1, 4, -1, -1).clone()
valid = torch.ones(2, 4, 3, device=device)
model = Tracker().to(device)
batch = (video, points, trajectory, valid)
```


### Cell 3 — Forward và baseline

Đọc shape/output trước khi train. Các số in ra chỉ minh họa phép tính và luồng dự đoán.

```python
model.eval()
with torch.no_grad():
    prediction = model(video, points)
print('Trajectory B x T x N x XY:', prediction.shape)
print('Stationary baseline loss:', (trajectory - points[:, None]).abs().mean().item())
```


### Cell 4 — Một bước cập nhật

Tính đúng loss của nhiệm vụ, backward và cập nhật tham số. Một bước không phải quy trình huấn luyện hoàn chỉnh. TENT tự thực hiện update bên trong adapt.

```python
model.train()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model, *batch)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
print('One-step training loss:', loss.detach().item())
```


### Cell 5 — Chẩn đoán và hướng nâng cấp

Stationary → feature correlation + soft coordinates → local iterative update và visibility head. Fixture frame lặp chỉ kiểm tra đường chạy, không đo tracking chuyển động.

```python
model.eval()
with torch.no_grad():
    prediction = model(video, points)
    normalized_l1 = ((prediction - trajectory).abs().sum(-1) * valid).sum() / valid.sum()
print('Fixture normalized coordinate L1:', normalized_l1.item())
print('No visibility head yet; this is not a full TAP-Vid average-Jaccard evaluator.')
```


## CV14 — Ghép ảnh và văn bản

Hai encoder đưa ảnh/text về cùng không gian. Batch B cặp tạo ma trận similarity B×B; đường chéo là positive, CE theo hàng và cột. ID ngẫu nhiên chỉ kiểm tra cơ chế; dữ liệu thật phải dùng tokenizer nhất quán.


### Cell 1 — Môi trường và seed

Dùng CPU để kiểm tra trước. Tìm repo root từ thư mục notebook, đặt seed và import PyTorch. Không tải pretrained weights.

```python
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'vision/curriculum').is_dir())
sys.path.insert(0, str(ROOT))
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device('cpu')  # Change to cuda after the CPU example works.
print('Device:', device)
```


### Cell 2 — Tạo batch và model

Hai encoder đưa ảnh/text về cùng không gian. Batch B cặp tạo ma trận similarity B×B; đường chéo là positive, CE theo hàng và cột. ID ngẫu nhiên chỉ kiểm tra cơ chế; dữ liệu thật phải dùng tokenizer nhất quán. Dữ liệu trong cell là fixture tổng hợp để bạn đọc được shape; thay bằng Dataset thật trước khi so metric.

```python
from vision.curriculum.cv14_open_vocab import DualEncoder, loss_fn
images = torch.rand(2, 3, 32, 32, device=device)
ids = torch.randint(0, 128, (2, 8), device=device)
mask = torch.ones(2, 8, dtype=torch.bool, device=device)
model = DualEncoder(vocab=128, d=32).to(device)
batch = (images, ids, mask)
```


### Cell 3 — Forward và baseline

Đọc shape/output trước khi train. Các số in ra chỉ minh họa phép tính và luồng dự đoán.

```python
model.eval()
with torch.no_grad():
    similarity = model(images, ids, mask)
print('Image-to-text matrix:', similarity.shape)
print('Positive target indices:', torch.arange(len(images)).tolist())
```


### Cell 4 — Một bước cập nhật

Tính đúng loss của nhiệm vụ, backward và cập nhật tham số. Một bước không phải quy trình huấn luyện hoàn chỉnh. TENT tự thực hiện update bên trong adapt.

```python
model.train()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model, *batch)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
print('One-step training loss:', loss.detach().item())
```


### Cell 5 — Chẩn đoán và hướng nâng cấp

Closed-set classifier → train dual encoder → load CLIP local → region-text alignment. Chưa có box head nên không phải open-vocabulary detector.

```python
model.eval()
with torch.no_grad():
    similarity = model(images, ids, mask)
print('Fixture retrieval choices:', similarity.argmax(-1).tolist())
print('Use load_clip(local_path) for permitted pretrained CLIP weights and its processor.')
print('A tiny randomly initialized dual encoder has no useful CLIP zero-shot ability.')
```


## CV15 — Dự đoán frame tương lai

CNN mã hóa mỗi frame, GRU tích lũy lịch sử, decoder dựng frame tiếp theo. Loss so output ở thời điểm t với frame t+1. Ví dụ teacher forcing, chưa có action hoặc rollout autoregressive.


### Cell 1 — Môi trường và seed

Dùng CPU để kiểm tra trước. Tìm repo root từ thư mục notebook, đặt seed và import PyTorch. Không tải pretrained weights.

```python
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'vision/curriculum').is_dir())
sys.path.insert(0, str(ROOT))
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device('cpu')  # Change to cuda after the CPU example works.
print('Device:', device)
```


### Cell 2 — Tạo batch và model

CNN mã hóa mỗi frame, GRU tích lũy lịch sử, decoder dựng frame tiếp theo. Loss so output ở thời điểm t với frame t+1. Ví dụ teacher forcing, chưa có action hoặc rollout autoregressive. Dữ liệu trong cell là fixture tổng hợp để bạn đọc được shape; thay bằng Dataset thật trước khi so metric.

```python
from vision.curriculum.cv15_world import World, loss_fn
video = torch.rand(2, 4, 3, 32, 32, device=device)
model = World(d=32).to(device)
batch = (video,)
```


### Cell 3 — Forward và baseline

Đọc shape/output trước khi train. Các số in ra chỉ minh họa phép tính và luồng dự đoán.

```python
model.eval()
with torch.no_grad():
    prediction = model(video[:, :-1])
print('Predicted future frames:', prediction.shape)
print('Aligned targets:', video[:, 1:].shape)
print('Repeat-last baseline L1:', F.l1_loss(video[:, :-1], video[:, 1:]).item())
```


### Cell 4 — Một bước cập nhật

Tính đúng loss của nhiệm vụ, backward và cập nhật tham số. Một bước không phải quy trình huấn luyện hoàn chỉnh. TENT tự thực hiện update bên trong adapt.

```python
model.train()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model, *batch)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
print('One-step training loss:', loss.detach().item())
```


### Cell 5 — Chẩn đoán và hướng nâng cấp

Repeat-last → CNN+GRU → spatial latent dynamics → action-conditioned model. Global average pooling làm mất cấu trúc không gian, là điểm cần nâng cấp.

```python
model.eval()
with torch.no_grad():
    prediction = model(video[:, :-1])
print('Fixture one-step L1:', F.l1_loss(prediction, video[:, 1:]).item())
print('A real study must also measure multi-step rollout stability and split by original video.')
```


## CV16 — Học quên có đối chứng

Bắt đầu từ source đã train, chia retain/forget và untouched eval. Mục tiêu minh họa giữ retain CE, gần reference trên retain, đẩy output forget về uniform. Không chứng minh xóa dữ liệu hay privacy.


### Cell 1 — Môi trường và seed

Dùng CPU để kiểm tra trước. Tìm repo root từ thư mục notebook, đặt seed và import PyTorch. Không tải pretrained weights.

```python
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'vision/curriculum').is_dir())
sys.path.insert(0, str(ROOT))
torch.manual_seed(42)
torch.set_num_threads(2)
device = torch.device('cpu')  # Change to cuda after the CPU example works.
print('Device:', device)
```


### Cell 2 — Tạo batch và model

Bắt đầu từ source đã train, chia retain/forget và untouched eval. Mục tiêu minh họa giữ retain CE, gần reference trên retain, đẩy output forget về uniform. Không chứng minh xóa dữ liệu hay privacy. Dữ liệu trong cell là fixture tổng hợp để bạn đọc được shape; thay bằng Dataset thật trước khi so metric.

```python
from copy import deepcopy
from vision.curriculum.cv16_unlearning import objective
retain_images = torch.rand(2, 3, 32, 32, device=device)
retain_labels = torch.tensor([0, 1], device=device)
forget_images = torch.rand(2, 3, 32, 32, device=device)
forget_labels = torch.tensor([1, 0], device=device)
model = nn.Sequential(nn.Flatten(), nn.Linear(3072, 2)).to(device)
reference = deepcopy(model).eval().requires_grad_(False)
batch = (retain_images, retain_labels, forget_images, forget_labels)
```


### Cell 3 — Forward và baseline

Đọc shape/output trước khi train. Các số in ra chỉ minh họa phép tính và luồng dự đoán.

```python
with torch.no_grad():
    source_probabilities = reference(forget_images).softmax(-1)
print('Initial forget probabilities:', source_probabilities.tolist())
print('Reference stays fixed:', all(not p.requires_grad for p in reference.parameters()))
```


### Cell 4 — Một bước cập nhật

Tính đúng loss của nhiệm vụ, backward và cập nhật tham số. Một bước không phải quy trình huấn luyện hoàn chỉnh. TENT tự thực hiện update bên trong adapt.

```python
model.train()
optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
loss = objective(model, reference, *batch)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
print('One-step training loss:', loss.detach().item())
```


### Cell 5 — Chẩn đoán và hướng nâng cấp

Unchanged source → retain-only fine-tune → retain/reference/uniform objective → so với retrain retain-only. forget_y đang không dùng trong objective này, có chủ đích.

```python
model.eval()
with torch.no_grad():
    retain_accuracy = (model(retain_images).argmax(-1) == retain_labels).float().mean()
    forget_confidence = model(forget_images).softmax(-1).max(-1).values.mean()
print('Fixture retain accuracy:', retain_accuracy.item())
print('Fixture forget confidence:', forget_confidence.item())
print('Gold control: retrain from scratch on retain only; uniform predictions do not establish deletion.')
```
