# Olympic AI — Vision research curriculum

16 topics, lightweight PyTorch reference implementations, and a progression from baseline to paper-inspired upgrades. These are educational implementations, not official reproductions or verified SOTA results. Default model weights are random; replace with permitted local checkpoints. No dataset is bundled. `smoke_test.py` uses synthetic data, tests CPU forward/loss/backward, and is not an accuracy benchmark.

## Start here

```bash
python -m vision.curriculum.smoke_test
```

Requirements: Python, torch, torchvision; optional transformers for library recipes. Match CUDA wheels to the competition environment. No internet needed for the default smoke test. Run from the repository root. `common.py` provides ResNet18 features, tokenization, cross-attention, TransformerEncoder, Dice and a BF16 train step. `library_recipes.py` provides Swin, ViT, Mask R-CNN, DeepLab, RAFT and local weight loading. Use training-set normalization appropriate for the supplied weights. The code uses compact shapes for readability; tune resolutions and batch sizes on the actual GPU.

```python
from vision.curriculum.common import train_step
from vision.curriculum.cv03_visual_context import VisualContext, loss_fn
import torch
model = VisualContext(mode='cross').cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
# loader yields support_images, support_masks, query_image, query_mask
for batch in loader:
    loss = train_step(model, opt, loss_fn, batch)
```

To use an offline torchvision ResNet checkpoint in a model with `model.enc`:

```python
from vision.curriculum.common import local_backbone
model.enc.net = local_backbone('/path/to/resnet18.pth')
```

For CV01 feature reconstruction, pass a **frozen pretrained** teacher, e.g. `SlotAutoencoder(feature_mode=True, teacher=local_backbone(path))`. Without that checkpoint the default feature branch is only a random-feature smoke-test control, not DINOSAUR.

## 01 — Object-centric learning (`cv01_slots.py`)
- Data: CLEVRTex / CLEVR. Input RGB [B,3,H,W]; train without masks. Output K masks (permutation invariant); evaluate FG-ARI, mIoU with Hungarian assignment. Do not use validation masks for training.
- B0: K-means over RGB+XY or frozen patch features. N1: `SlotAutoencoder(feature_mode=False)` reconstructs downsampled RGB via competitive attention + GRU. N2: `feature_mode=True` reconstructs frozen features, DINOSAUR-inspired; use a permitted pretrained teacher. N3: multi-layer features and slot overlap merging, not implemented.
- Read: [Slot Attention](https://arxiv.org/abs/2006.15055), [DINOSAUR](https://arxiv.org/abs/2209.14860), [SPOT](https://arxiv.org/abs/2312.00648), [Slot Merging 2026](https://arxiv.org/abs/2603.11246), [MUFASA 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Bock_MUFASA_A_Multi-Layer_Framework_for_Slot_Attention_CVPR_2026_paper.html).
- Core: attention softmax over slots, then normalize over spatial tokens. Watch for background collapse and redundant slots. Evaluate with fixed seeds.

## 02 — Abstract visual reasoning (`cv02_reasoning.py`)
- Data: RAVEN / I-RAVEN, matrix completion; context [B,8,3,H,W], candidates [B,A,3,H,W], answer index [B]. Metric accuracy, including held-out rules. Split by problem templates where possible.
- B0: `Reasoner(mode='baseline')` CNN embeddings + MLP. N1: `relation` all pairwise relations. N2: `transformer` joint reasoning over eight context panels and candidate. N3: explicit row/column rule supervision or symbolic attribute inference (exercise).
- Read: [Relation Networks](https://arxiv.org/abs/1706.01427), [RAVEN](https://arxiv.org/abs/1903.02741), [Bongard-LOGO](https://arxiv.org/abs/2010.00763).
- Core: candidate-specific processing; avoid answer position leakage and shortcuts from dataset generation. Randomize answer order with labels remapped.

## 03 — Visual in-context learning (`cv03_visual_context.py`)
- Data: Pascal-5i / COCO-20i or synthetic image transformations. Support images [B,S,3,H,W], support binary masks [B,S,1,H,W], query image and target mask. Evaluate novel-class mIoU, avoid train/test class overlap. This implementation is few-shot binary segmentation, not arbitrary task Painter.
- B0: `VisualContext(mode='prototype')`: masked average pooling + cosine foreground/background. N1: `cross` label-conditioned support tokens + query-to-support attention. N2: image/mask grid completion with masked ViT (exercise).
- Read: [PANet](https://arxiv.org/abs/1908.06391), [Visual Prompting via Image Inpainting](https://arxiv.org/abs/2209.00647), [Painter](https://arxiv.org/abs/2212.02499), [SegGPT](https://arxiv.org/abs/2304.03284), [Visual In-Context Prompting](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Visual_In-Context_Prompting_CVPR_2024_paper.html).
- Core: support/query episodes; cross-attention is reusable for guided SR and reference segmentation.

## 04 — Compositional generalization (`cv04_composition.py`)
- Data: MIT-States, UT-Zappos, C-GQA, CLEVR-CoGenT. x image, pairs [P,2] attribute/object IDs, target pair index. Train with seen pairs only; evaluate seen/unseen accuracy, harmonic mean and bias-sweep AUC. All primitive classes needed in test should appear in train.
- B0: flat pair classifier (cannot naturally predict unseen pairs). N1: `Composition(mode='factorized')` sums attribute and object logits. N2: `joint` learned compositional embeddings + auxiliary primitive CE. N3: cross-modal prompts/flow composition if permitted CLIP weights exist (exercise).
- Read: [CSP](https://arxiv.org/abs/2204.03574), [Troika](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_Troika_Multi-Path_Cross-Modal_Traction_for_Compositional_Zero-Shot_Learning_CVPR_2024_paper.html), [FlowComposer 2026](https://hkust-longgroup.github.io/FlowComposer/), [CZSL survey](https://arxiv.org/abs/2510.11106).
- Core: output all candidate compositions at test; keep calibration and generalized split consistent.

## 05 — Subtle visual computing / rPPG (`cv05_subtle.py`)
- Data: UBFC-rPPG, PURE, or permitted competition data. Cropped skin ROI videos [B,T,3,H,W], ground truth aligned waveform [B,T]. Metric HR MAE/RMSE, waveform correlation. Use subject-disjoint splits and actual FPS; use sufficiently long windows (e.g. 10 s).
- B0: average green trace + FFT. B1: `chrom` chrominance signal (simplified, no full windowed filtering). N1: `PulseNet` small 3D CNN + correlation/frequency losses. N2: temporal difference attention / physiological augmentation (exercise).
- Read: [PhysNet](https://arxiv.org/abs/1905.02419), [PhysFormer](https://arxiv.org/abs/2111.12082), [EfficientPhys](https://arxiv.org/abs/2110.04447), [SVC 2026 challenge](https://openaccess.thecvf.com/content/CVPR2026W/SVC/html/Zhu_SVC_2026_the_Second_Multimodal_Deception_Detection_Challenge_and_the_CVPRW_2026_paper.html).
- Core: frequency-domain supervision, avoid temporal augmentations that change label frequency unless labels are transformed. This is a research exercise, not clinical measurement.

## 06 — Test-time adaptation (`cv06_adapt.py`)
- Data: CIFAR-10-C / ImageNet-C; train clean, test corruptions. Evaluate mean corruption accuracy. Only adapt on test images if rules explicitly allow; never use test labels. Compare fixed source and reset between domains as protocol specifies.
- B0: frozen source model. B1: BN statistics adaptation. N1: `configure_tent` + `adapt` update BN affine parameters via entropy. N2: entropy threshold filtering inspired by EATA (not full EATA). N3: teacher-student or robust sharpness adaptation (exercise).
- Read: [TENT](https://arxiv.org/abs/2006.10726), [EATA](https://arxiv.org/abs/2204.02610), [SAR](https://arxiv.org/abs/2302.12400).
- Core: adaptation can collapse on class-imbalanced batches. Keep immutable source checkpoint and validate protocol.

## 07 — Visual tokenization (`cv07_tokenization.py`)
- Data: CIFAR-10 or image patches. Input RGB sizes divisible by 4. Metric reconstruction PSNR/SSIM, codebook perplexity, downstream accuracy.
- B0: continuous convolutional autoencoder. N1: `VQVAE`, nearest codebook vector + straight-through gradient + codebook/commitment loss. N2: train GPT over discrete token IDs via `small_token_lm` in library_recipes (flatten raster tokens, next-token CE). N3: perceptual reconstruction + residual quantization (exercise).
- Read: [VQ-VAE](https://arxiv.org/abs/1711.00937), [VQGAN](https://arxiv.org/abs/2012.09841), [MaskGIT](https://arxiv.org/abs/2202.04200), [AToken 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_AToken_A_Unified_Tokenizer_for_Vision_CVPR_2026_paper.html).
- Core: detect dead codebook entries; reconstruction alone does not guarantee semantic representation.

## 08 — Flow matching (`cv08_flow.py`)
- Data: MNIST/CIFAR or image restoration. Input RGB scaled to [-1,1]. Metric task dependent: FID on sufficiently large generation set, or PSNR for conditional restoration. Avoid comparing FID on tiny sample sets.
- B0: deterministic autoencoder for reconstruction (different objective). N1: `Velocity` learns straight-line Gaussian-to-data velocity. N2: replace CNN with time-conditioned U-Net, add condition channels and train paired degraded/clean flow (exercise). N3: improved ODE solvers and transport couplings (exercise).
- Read: [Flow Matching](https://arxiv.org/abs/2210.02747), [Rectified Flow](https://arxiv.org/abs/2209.03003), [DiT](https://arxiv.org/abs/2212.09748).
- Core: t=0 noise, t=1 data, Euler update positive dt. Use `sample`; don't confuse velocity prediction with direct image regression.

## 09 — AI-generated image detection (`cv09_forensics.py`)
- Data: GenImage or multi-generator corpus. RGB input and binary labels. AUROC, balanced accuracy, unseen-generator performance. Split by generator and source image, avoid watermark/compression leakage.
- B0: `Forensics(mode='rgb')`. N1: `fusion` RGB + FFT log amplitude branch. N2: patch token classifier and multi-scale crop ensemble (exercise).
- Read: [CNN-generated images are surprisingly easy to spot](https://arxiv.org/abs/1912.11035), [UniversalFakeDetect](https://arxiv.org/abs/2302.10174), [SAFE 2026](https://www.safeworkshop.org/cvpr-2026/).
- Core: augment JPEG/resizing symmetrically; test on unseen generator before trusting accuracy.

## 10 — All-in-one restoration (`cv10_restoration.py`)
- Data: clean/degraded aligned pairs with noise, blur, rain, etc. Metric PSNR/SSIM by degradation. Train random crops with paired transforms.
- B0: identity. N1: `Restoration(condition=False)` residual CNN. N2: `condition=True` learned feature modulation (FiLM-inspired, not PromptIR). N3: multi-scale U-Net / Restormer / degradation prompts (exercise).
- Read: [Restormer](https://arxiv.org/abs/2111.09881), [AirNet](https://arxiv.org/abs/2203.09368), [PromptIR](https://arxiv.org/abs/2306.13090), [LoViF 2026](https://lovif-cvpr2026-workshop.github.io/).
- Core: output residual, condition model on degradation, compare per corruption not only pooled score.

## 11 — Dense correspondence (`cv11_matching.py`)
- Data: SPair-71k, HPatches. Inputs paired images and flattened target feature-cell indices [B,Nsource], -100 for unknown. Metric PCK / homography error. Transform labels along with geometric augmentation.
- B0: frozen features + cosine matching. N1: `Matcher(context=False)` supervised correlation CE. N2: `context=True` cross attention. N3: local refinement on high-resolution features, mutual consistency (exercise). `soft_coordinates` gives expected (x,y) feature-grid coordinates.
- Read: [SuperGlue](https://arxiv.org/abs/1911.11763), [LoFTR](https://arxiv.org/abs/2104.00680), [GMFlow](https://arxiv.org/abs/2111.13680), [RoMa](https://arxiv.org/abs/2305.15404).
- Core: global correlation memory grows quadratically with spatial tokens; coarse-to-fine is a quality and compute strategy.

## 12 — Amodal segmentation (`cv12_amodal.py`)
- Data: KINS, COCOA, D2SA. RGB, supplied visible mask, complete mask [B,1,H,W]. Metric full and invisible-region IoU. If visible masks aren't given, use a detector/segmenter first.
- B0: visible mask itself. N1: `Amodal` small encoder-decoder predicts invisible region, union via `predict`. N2: shape prototypes, multi-head masks, synthetic copy-paste occlusion (exercise).
- Read: [AISFormer](https://arxiv.org/abs/2210.06323), [ShapeFormer](https://arxiv.org/abs/2403.11376), [pix2gestalt](https://arxiv.org/abs/2401.14398).
- Core: enforce full mask contains visible mask; report occlusion-specific metric.

## 13 — Point tracking (`cv13_tracking.py`)
- Data: TAP-Vid or synthetic translated images. Video [B,T,3,H,W], points [B,N,2] in first frame, trajectories [B,T,N,2] normalized [-1,1], validity [B,T,N]. Metric average Jaccard / position accuracy / occlusion accuracy; current code predicts coordinates only.
- B0: stationary point. N1: `Tracker` first-frame query feature + all-frame correlation and soft coordinates. N2: local iterative update / forward-backward consistency / visibility head (exercise). Use torchvision RAFT for flow baseline if weights allowed.
- Read: [RAFT](https://arxiv.org/abs/2003.12039), [TAP-Vid](https://arxiv.org/abs/2211.03726), [TAPIR](https://arxiv.org/abs/2306.08637), [CoTracker](https://arxiv.org/abs/2307.07635).
- Core: `grid_sample` coordinate convention, occlusion, drift; split by video.

## 14 — Open-vocabulary perception (`cv14_open_vocab.py`)
- Data: paired image/text or class descriptions, held-out concepts. Metric zero-shot class accuracy / retrieval recall; detection needs additional localization head and AP.
- B0: ordinary closed-set classifier. N1: `DualEncoder` trained with symmetric contrastive CE, local tokenizer required. N2: `load_clip(path)` for permitted local CLIP. N3: region-level text alignment and open-vocabulary detection (exercise).
- Read: [CLIP](https://arxiv.org/abs/2103.00020), [OWL-ViT](https://arxiv.org/abs/2205.06230), [Grounding DINO](https://arxiv.org/abs/2303.05499).
- Core: training a small dual encoder does not reproduce CLIP zero-shot ability; duplicate captions are false negatives unless handled.

## 15 — World models (`cv15_world.py`)
- Data: Moving MNIST (convert to 3 channels), simple trajectories. Metric future-frame MSE/SSIM and long-horizon prediction; current module is teacher-forced one-step model, no actions.
- B0: repeat last frame. N1: `World` encoder + GRU + image decoder. N2: predict spatial latent tokens rather than pooled vector, causal temporal Transformer and rollout (exercise). N3: action-conditioned latent dynamics (exercise).
- Read: [World Models](https://arxiv.org/abs/1803.10122), [I-JEPA](https://arxiv.org/abs/2301.08243), [V-JEPA](https://arxiv.org/abs/2404.08471).
- Core: one-step quality differs from rollout stability; avoid train/test overlap of videos.

## 16 — Model unlearning (`cv16_unlearning.py`)
- Data: classification corpus divided into forget, retain, untouched evaluation. Metric retain accuracy, forget behavior, comparison to retrain on retain, and privacy attack metrics where relevant.
- B0: unchanged source; gold reference retrain on retain only. N1: retain-only fine-tuning. N2: `objective` retain CE + reference consistency + uniform prediction on forget samples. N3: more targeted parameter editing, audit against reference retraining (exercise).
- Read: [Machine Unlearning](https://arxiv.org/abs/1912.03817), [Selective Synaptic Dampening](https://arxiv.org/abs/2308.07707).
- Core: uniform forget predictions do not demonstrate privacy or certified deletion. The example is only a pedagogical objective.

## Competition protocol

1. Read metric and input/output constraints. Make a valid baseline and validate submission formatting first.
2. Use identical validation split and training recipe for each ablation. For images use group/subject/video splits as needed.
3. Start with permitted pretrained models; compare frozen features versus fine-tune. BF16 on supported hardware. Cache features when frozen, use torch DataLoader pin_memory and workers, benchmark actual throughput.
4. Train one major upgrade, not five simultaneous changes. Keep top checkpoint and log metric per domain/class. Budget last 30–45 minutes for inference, validation and ensemble when permitted.
5. No test-time fitting, external weights, labels or pseudo-labels unless contest rules allow. Do not assume half-H200 implies a specific MIG profile.

## Limitations
This is a first study pack, not 16 full dataset pipelines. Dataset adapters, exact official metric implementations, full training runs, and exercises marked above remain to be completed. Code is small intentionally; none of the advanced variants is claimed to reproduce published SOTA. The optional transformers loaders need local model files or installed transformers and are not part of the synthetic smoke test.
