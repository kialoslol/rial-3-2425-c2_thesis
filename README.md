# UAV Defect Detection

Semantic segmentation (and one object-detection variant) of structural
defects — cracks, spalling, delamination, moisture — from **RGB + Infrared
(IR) UAV imagery**, captured on drone bridge/structure inspections.

Every model variant lives in its own folder under `models/`, each with its
own architecture, training pipeline, and checkpoints, sharing common
training/data infrastructure from `common/`. See **Project Structure** below,
and each `models/<name>/README.md` for full detail on that model.

---

## Class Map

| ID | Class | Notes |
|---|---|---|
| 0 | Background | Dominant class; down-weighted in loss |
| 1 | Crack | Surface fractures |
| 2 | Spall | Concrete spalling |
| 3 | Delamination | Reserved (no annotations in current dataset) |
| 4 | Moisture | Moisture infiltration; severe class imbalance (<0.1% of pixels) |

---

## Models

| Folder | Task | Backbone | Fusion | Status |
|---|---|---|---|---|
| [`models/early_fusion_v2`](models/early_fusion_v2/README.md) | Segmentation | MobileViTv2-100 + ASPP | Early (4ch→3ch) | **Primary model** — trained 200 epochs, see its README for full results |
| [`models/early_fusion_v1`](models/early_fusion_v1/README.md) | Segmentation | MobileViT-S | Early (4ch→3ch) | Earlier baseline, superseded by v2 |
| [`models/CNN_VIT_SELF`](models/CNN_VIT_SELF/README.md) | Segmentation | Custom, from scratch (no pretrained backbone) | Early (native 4ch) | New — verified end-to-end (forward/backward + one real training epoch), not yet trained to convergence |
| [`models/mobilenet_v4`](models/mobilenet_v4/README.md) | Segmentation | MobileNetV4-Conv-Medium | Early (4ch→3ch) | New — verified end-to-end (pretrained download/load + one real training epoch), not yet trained to convergence |
| [`models/late_fusion_v1`](models/late_fusion_v1/README.md) | Segmentation | Dual MobileViT-S | Late (decoder attention) | Model implemented; training pipeline newly added, not yet run end-to-end |
| [`models/late_fusion_v2`](models/late_fusion_v2/README.md) | Segmentation | Dual MobileViTv2-100 | Late (decoder attention) | Model implemented; training pipeline newly added, not yet run end-to-end |
| [`models/single_modal_v1`](models/single_modal_v1/README.md) | Segmentation | MobileViT-S | None (RGB only) | Known dataset/model channel mismatch — see its README |
| [`models/single_modal_v2`](models/single_modal_v2/README.md) | Segmentation | MobileViTv2-100 | None (RGB only) | Same known mismatch as v1 |
| [`models/yolov11_defect_detection`](models/yolov11_defect_detection/README.md) | Object detection | YOLOv11 | n/a | Single-class ("defect") bounding-box detector |
| [`models/fastvit_sa12`](models/fastvit_sa12/README.md) | Segmentation | FastViT-SA12 | Early (4ch→3ch) | New — trains unfused, reparameterizes on export only, not yet trained to convergence |
| [`models/efficientformerv2_s2`](models/efficientformerv2_s2/README.md) | Segmentation | EfficientFormerV2-S2 | Early (4ch→3ch) | New — trains at native 224px (resolution exception, see its README), not yet trained to convergence |
| [`models/edgenext_small`](models/edgenext_small/README.md) | Segmentation | EdgeNeXt-Small | Early (4ch→3ch) | New — not yet trained to convergence |
| [`models/swiftformer_l1`](models/swiftformer_l1/README.md) | Segmentation | SwiftFormer-L1 | Early (4ch→3ch) | New — not yet trained to convergence |
| [`models/repvit_m1_1`](models/repvit_m1_1/README.md) | Segmentation | RepViT-M1.1 | Early (4ch→3ch) | New — trains unfused, reparameterizes on export only, not yet trained to convergence |
| [`models/efficientnet_b0`](models/efficientnet_b0/README.md) | Segmentation | EfficientNet-B0 (pure-CNN baseline) | Early (4ch→3ch) | New — not yet trained to convergence |
| [`models/deit_small`](models/deit_small/README.md) | Segmentation | DeiT-Small/patch16/224 (pure-ViT baseline) | Early (4ch→3ch) | New — custom 4-stage decoder (stride-16 backbone), not yet trained to convergence |

`models/CNN_VIT_SELF` is the only model here that isn't built on a `timm`
pretrained backbone — its CNN stem, CNN stages, and self-attention blocks
are all defined and trained from scratch. See its README for why that also
means it needs no 4→3 channel-projection layer.

`models/mobilenet_v4`'s backbone choice is documented in
[`docs/MobileRankings_TIMM.docx`](docs/MobileRankings_TIMM.docx), a ranking
of 222 mobile-class TIMM models by ImageNet accuracy.

"Early" fusion projects the 4-channel RGB+IR TIFF down to 3 channels with a
learnable 1x1 conv before a single backbone. "Late" fusion runs RGB and IR
through two independent backbones and fuses their features at the decoder
via learned modality attention — useful when the two modalities carry
complementary, not just redundant, information (RGB = surface texture,
IR = subsurface thermal anomalies).

### Best result so far (Early Fusion v2, epoch 200)

| Metric | Value |
|---|---|
| Mean IoU | 47.12% |
| Mean Dice | 58.95% |
| Macro F1 | 52.82% |

Full per-class breakdown and training curves: [`models/early_fusion_v2/README.md`](models/early_fusion_v2/README.md).

---

## Project Structure

```
THESIS/
├── common/                      # Shared trainer, metrics, dataset loaders, viz — used by every model
│   ├── heads.py                 #   MobileViTSegmentationHead (shared decoder head)
│   ├── trainer.py                #   DefectDetectionTrainer, DefectDetectionMetrics, checkpoint helpers
│   ├── losses.py                 #   FocalDiceLoss (shared by early_fusion_v2 and CNN_VIT_SELF)
│   ├── dataset.py                #   COCOSegmentationDataset, MultiModalDataset
│   ├── dataset_utils.py          #   CVAT/Pascal-VOC XML -> COCO JSON converters, class-weight/oversampling helpers
│   └── visualization.py          #   Training curves, confusion matrix plots
│
├── models/                      # One folder per model/variation
│   ├── early_fusion_v2/          #   Primary model — see table above
│   ├── early_fusion_v1/
│   ├── CNN_VIT_SELF/              #   From-scratch CNN + self-attention hybrid (no pretrained backbone)
│   ├── mobilenet_v4/               #   MobileNetV4-Conv-Medium backbone (see docs/MobileRankings_TIMM.docx)
│   ├── late_fusion_v1/
│   ├── late_fusion_v2/
│   ├── single_modal_v1/
│   ├── single_modal_v2/
│   ├── yolov11_defect_detection/ #   Object detection (separate from the segmentation models)
│   ├── fastvit_sa12/              #   FastViT-SA12 (trains unfused, reparameterize() on export only)
│   ├── efficientformerv2_s2/      #   EfficientFormerV2-S2 (native 224px — resolution exception, see its README)
│   ├── edgenext_small/            #   EdgeNeXt-Small
│   ├── swiftformer_l1/            #   SwiftFormer-L1
│   ├── repvit_m1_1/               #   RepViT-M1.1 (trains unfused, reparameterize() on export only)
│   ├── efficientnet_b0/           #   EfficientNet-B0 — pure-CNN baseline
│   ├── deit_small/                #   DeiT-Small/patch16/224 — pure-ViT baseline (custom 4-stage decoder)
│   └── <name>/model.py, train.py, checkpoints/, README.md
│
├── docs/                         # Reference documents (not code)
│   └── MobileRankings_TIMM.docx  #   TIMM mobile-model ranking behind the mobilenet_v4 backbone choice
│
├── scripts/                     # Repo-wide utility scripts (not tied to one model)
│   ├── channelstack.py           #   Fuse RGB + IR into 4-channel TIFFs
│   ├── prepare_dataset.py        #   Legacy Pascal-VOC dataset preparation
│   ├── quickstart.py             #   Legacy end-to-end demo (predates the CVAT/config.yaml flow)
│   ├── test_installation.py      #   Verify environment + imports
│   ├── count_classes.py          #   Count annotation instances per class
│   └── rename_files.py           #   Batch-rename files in a folder
│
├── dashboard.py                  # Streamlit inference dashboard (loads any model's checkpoints)
├── config.yaml                   # Shared dataset paths + training hyperparameters
│
├── RGB_FITTED/ IRT_FITTED/ FUSED/ ANNOTATED_DATA/   # Shared source imagery + CVAT annotations
└── dataset/                      # COCO-formatted dataset (shared across models)
    ├── instances_train.json
    └── instances_val.json
```

---

## Installation

**Requirements:** Python 3.8+, CUDA 11.0+ (recommended)

```bash
pip install -r requirements.txt
```

Verify everything (packages, GPU, config, a model forward pass):
```bash
python scripts/test_installation.py
```

See `SETUP.md` for a more detailed walkthrough, and its Troubleshooting
section if `pretrained=True` model downloads fail with an SSL error (common
on university/corporate networks that TLS-intercept — `common/__init__.py`
works around this automatically via `truststore` as long as it's installed).

---

## Dataset Setup

### 1 — Fuse RGB + IR into 4-channel TIFFs (only needed for new raw data)

```bash
python scripts/channelstack.py
```

Writes `.tiff` files containing `[R, G, B, IR]` channels. The repo already
ships a fused dataset under `ANNOTATED_DATA/`.

### 2 — Convert CVAT XML annotations to COCO JSON

This runs automatically on first training via `CVATXMLToCOCOConverter`
(`common/dataset_utils.py`), driven by every `models/<name>/train.py`. To run
it manually:

```python
from common.dataset_utils import CVATXMLToCOCOConverter

converter = CVATXMLToCOCOConverter("path/to/annotations.xml")
converter.convert(
    output_train="dataset/instances_train.json",
    output_val="dataset/instances_val.json",
    train_ratio=0.8,
)
```

### 3 — Dataset Layout

```
ANNOTATED_DATA/           # 4-channel fused TIFFs, in per-batch subfolders
├── fuse_Binondo/
├── FUSED/
└── ...
dataset/
├── instances_train.json
└── instances_val.json
RGB_FITTED/               # RGB images (for the late-fusion models)
IRT_FITTED/                # IR thermal images (for the late-fusion models)
```

See `DATASET_FORMAT_GUIDE.md` for the full annotation format spec.

---

## Configuration (`config.yaml`)

One shared config drives dataset paths and training hyperparameters for
every model. Each `models/<name>/train.py` overrides the checkpoint
directory to its own `models/<name>/checkpoints/` folder, so
`checkpoint.save_dir` in `config.yaml` is only a fallback — training runs
never collide with each other on disk.

```yaml
dataset:
  fused_dir:             "ANNOTATED_DATA"     # 4-channel TIFFs (early fusion / single-modal)
  rgb_dir:                "RGB_FITTED"         # RGB images (late fusion)
  irt_dir:                "IRT_FITTED"         # IR images (late fusion)
  annotation_dir:         "dataset"
  train_annotation_file:  "dataset/instances_train.json"
  val_annotation_file:    "dataset/instances_val.json"
  xml_annotation:         "ANNOTATED_DATA/fifthmerged_output.xml"
  num_classes: 5
  image_size:  384
  train_ratio: 0.8

training:
  batch_size:               6
  num_epochs:               300
  learning_rate:            0.0003
  early_stopping_patience:  60
```

---

## Training

Every model trains from the repo root the same way — each `train.py`
resolves `config.yaml` and its own `checkpoints/` folder relative to its own
file location, so the current working directory doesn't matter:

```bash
python models/early_fusion_v2/train.py     # primary model
python models/early_fusion_v1/train.py
python models/CNN_VIT_SELF/train.py        # from-scratch CNN + self-attention hybrid
python models/mobilenet_v4/train.py        # MobileNetV4-Conv-Medium backbone
python models/late_fusion_v1/train.py
python models/late_fusion_v2/train.py
python models/single_modal_v1/train.py     # see README — known dataset/model mismatch
python models/single_modal_v2/train.py     # see README — known dataset/model mismatch
```

Each pipeline will:
1. Convert CVAT XML → COCO JSON (skipped if it already exists)
2. Train with that model's loss/scheduler (see its README for specifics)
3. Save the best checkpoint(s) to `models/<name>/checkpoints/`
4. Log final per-class metrics

---

## Visualization

After training, generate plots from a saved `training_history.json`:

```bash
python common/visualization.py models/early_fusion_v2/checkpoints/training_history.json
```

Produces two files in `./visualizations/` (relative to wherever you run the
command from):

| File | Contents |
|---|---|
| `training_history.png` | 6-panel grid: Loss, Accuracy, IoU, Dice, F1, mAP — Train vs Val per epoch, best epoch annotated |
| `confusion_matrix.png` | Pixel-count heatmap (normalized by row) + per-class Precision/Recall/F1 bar chart |

---

## Inference / Dashboard

```bash
streamlit run dashboard.py
```

The dashboard scans `models/*/checkpoints/*.pth`, picks the right
architecture from the checkpoint filename prefix (`early_fusion`,
`_v2_` for late/single-modal v2, otherwise late-fusion v1), and supports
RGB+IR, RGB-only, or IR-only inference with missing-modality zero-filling.

For programmatic single-image inference, see the "Usage" section in each
model's README, e.g. [`models/early_fusion_v2/README.md`](models/early_fusion_v2/README.md#inference-with-moisture-confidence-thresholding).

---

## Data Augmentation

Applied to the training split only (`common/dataset.py`, via albumentations
so all geometric transforms stay pixel-aligned across RGB/IR/mask):

| Type | Transform |
|---|---|
| Geometric | Horizontal/vertical flip, 90° rotation, affine (translate/scale/rotate), elastic transform, grid distortion |
| Color | Brightness/contrast, color jitter, CLAHE, sharpen |
| Blur / Noise | Gaussian blur, Gaussian noise |
| Copy-paste | Crack/moisture defect patches pasted onto other images (early-fusion dataset only) |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'common'` or `'models'`**
- Run from the repo root: `python models/<name>/train.py`. Every
  entry-point script inserts the repo root onto `sys.path` itself.

**`SSLCertVerificationError` / `LocalEntryNotFoundError` when a model loads `pretrained=True`**
- Some networks (university/corporate proxies, some antivirus software)
  TLS-intercept HTTPS traffic with their own certificate. Windows and
  `curl` trust it (it's in the OS store); Python's `certifi` bundle doesn't,
  so `timm`'s Hugging Face Hub download fails. `common/__init__.py` already
  works around this via `truststore.inject_into_ssl()` — make sure
  `truststore` is installed (`pip install -r requirements.txt`) and that
  you're importing something from `common`/`models` before the download
  happens (every `train.py` does this automatically).

**Out of memory (OOM)**
- Reduce `training.batch_size` or `dataset.image_size` in `config.yaml`.

**Moisture IoU near zero**
- Severe class imbalance (<0.1% of pixels). `early_fusion_v2` already
  applies a x5 weight boost, moisture oversampling, and inference-time
  confidence thresholding — see its README's "Note on Moisture".

**COCO JSON missing / empty**
- Delete the (likely empty) `dataset/instances_*.json` and re-run any
  `train.py` — it regenerates them from `ANNOTATED_DATA/*.xml`
  automatically.

**Slow training on CPU**
- Confirm CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce `num_workers` in the `DataLoader` calls if you hit deadlocks on Windows.

---

## References

- Mehta & Rastegari — [MobileViT (2021)](https://arxiv.org/abs/2110.02178)
- Mehta & Rastegari — [Separable Self-Attention for MobileViTv2 (2022)](https://arxiv.org/abs/2206.02680)
- Lin et al. — [Focal Loss (RetinaNet, 2017)](https://arxiv.org/abs/1708.02002)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [timm — PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
