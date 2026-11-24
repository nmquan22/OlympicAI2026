# CV01 — Slot Attention, giải thích từng cell

Đánh số cell code; markdown đứng trước mỗi cell giải thích mục tiêu.


## Cell 1 — Import và seed

Dùng torchvision làm encoder; chỉ khối Slot Attention và decoder nhỏ cần viết riêng.

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

from vision.curriculum.common import Encoder, tokens
from vision.curriculum.cv01_slots import SlotAttention, SlotAutoencoder
```


## Cell 2 — Batch ảnh

B=2 ảnh, 3 kênh, 64×64. Vẽ hai hình chữ nhật màu để đầu vào dễ nhìn. Không dùng mask giám sát để train.

```python
images = torch.full((2, 3, 64, 64), 0.1, device=device)
images[:, 0, 8:28, 8:28] = 0.9
images[:, 1, 32:54, 34:56] = 0.9
print('RGB images:', images.shape)
```


## Cell 3 — Encoder thư viện

ResNet18 layer2 giảm mỗi chiều 8 lần: 64×64 → 8×8. Projection 1×1 đưa 128 kênh về d=64. weights=None nên chưa có ngữ nghĩa pretrained.

```python
encoder = Encoder(d=64).to(device).eval()
with torch.no_grad():
    features = encoder(images)
print('Feature map B x d x h x w:', features.shape)
```


## Cell 4 — Đổi feature map thành tokens

Flatten trục h,w rồi transpose. N=8×8=64 tokens, mỗi token có 64 số. Đây là đổi cách bố trí tensor, không thêm tham số.

```python
patch_tokens = tokens(features)
print('Tokens B x N x d:', patch_tokens.shape)
```


## Cell 5 — Slot Attention

K=3 vector đại diện, cùng cạnh tranh nhận các token. Q đến từ slots, K/V đến từ patch tokens; lặp attention, GRU và MLP ba bước.

```python
slot_module = SlotAttention(d=64, k=3, steps=3).to(device).eval()
with torch.no_grad():
    slots, attention = slot_module(patch_tokens)
print('Slots B x K x d:', slots.shape)
print('Attention B x K x N:', attention.shape)
```


## Cell 6 — Hai chiều chuẩn hóa attention

softmax(dim=1) làm mỗi patch phân phối cho K slots. Sau đó chia tổng theo N để lấy weighted average cập nhật mỗi slot. Attention không phải decoder mask.

```python
slot_competition_sum = attention.sum(dim=1)
weights_for_update = attention / attention.sum(dim=-1, keepdim=True)
print('Per-patch slot probability sum:', slot_competition_sum.min().item(), slot_competition_sum.max().item())
print('Per-slot spatial weight sum:', weights_for_update.sum(dim=-1))
```


## Cell 7 — Ghép encoder, slots và decoder

Mỗi slot ghép tọa độ x,y rồi dự đoán RGB và mask logit ở từng vị trí. Softmax qua slots tạo mask; cộng RGB theo trọng số mask để dựng ảnh.

```python
model = SlotAutoencoder(d=64, k=3, feature_mode=False).to(device)
model.eval()
with torch.no_grad():
    prediction, target, masks = model(images)
print('Reconstruction:', prediction.shape)
print('Downsampled RGB target:', target.shape)
print('Decoder masks B x K x h x w:', masks.shape)
```


## Cell 8 — Loss và một bước train

MSE chỉ so ảnh dựng lại với RGB gốc đã downsample; không có nhãn object. Gradient đi qua decoder, slots và encoder. Một bước chưa tạo object masks đáng tin.

```python
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
optimizer.zero_grad(set_to_none=True)
prediction, target, masks = model(images)
loss = F.mse_loss(prediction, target)
loss.backward()
slot_gradient = model.slot.q.weight.grad.norm().item()
optimizer.step()
print('RGB reconstruction MSE:', loss.detach().item())
print('Slot query gradient norm:', slot_gradient)
```


## Cell 9 — Suy luận và mask

eval() không tắt việc khởi tạo slot ngẫu nhiên. Đặt lại seed nếu cần so đúng cùng mẫu. argmax tạo phân nhóm; slot ID không phải semantic class ID.

```python
model.eval()
torch.manual_seed(42)
with torch.no_grad():
    prediction, target, masks = model(images)
    assignment = masks.argmax(dim=1)
print('Partition map:', assignment.shape)
print('Mask-sum error:', (masks.sum(dim=1) - 1).abs().max().item())
print('Fixture reconstruction MSE:', F.mse_loss(prediction, target).item())
```


## Cell 10 — Baseline K-means RGB+XY

Control rẻ không cần neural training: cluster màu và vị trí. Nhãn cluster chỉ là ID nhóm; so ARI hoặc ghép Hungarian nếu có GT masks trên validation.

```python
from sklearn.cluster import KMeans
low_rgb = F.interpolate(images[:1], size=(8, 8), mode='bilinear', align_corners=False)
yy, xx = torch.meshgrid(torch.linspace(-1, 1, 8, device=device),
                        torch.linspace(-1, 1, 8, device=device), indexing='ij')
xy = torch.stack((xx, yy), dim=-1).reshape(64, 2)
color_and_position = torch.cat((tokens(low_rgb)[0], 0.25 * xy), dim=-1)
cluster_map = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(color_and_position.cpu().numpy()).reshape(8, 8)
print('K-means partition:', cluster_map.shape)
print('Cluster counts:', torch.bincount(torch.tensor(cluster_map).flatten()).tolist())
```


## Cell 11 — Nhìn ảnh, reconstruction và từng slot

Ghép ảnh gốc, RGB dựng lại và các mask thành một hàng. Notebook hiển thị ảnh PIL ở dòng cuối. Noise và chia vùng sai sau một update là điều bình thường.

```python
from PIL import Image
input_preview = F.interpolate(images[:1], (64, 64), mode='nearest')
reconstruction_preview = F.interpolate(prediction[:1].clamp(0, 1), (64, 64), mode='nearest')
mask_previews = F.interpolate(masks[:1].transpose(0, 1), (64, 64), mode='nearest').repeat(1, 3, 1, 1)
previews = torch.cat((input_preview, reconstruction_preview, mask_previews), dim=0)
strip = torch.cat(tuple(previews), dim=-1).permute(1, 2, 0)
preview = Image.fromarray((strip.cpu().numpy() * 255).astype('uint8'))
preview
```
