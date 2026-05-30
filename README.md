# UAV Defect Detection — MobileViTv2 Early Fusion

Semantic segmentation of structural defects from **RGB + Infrared (IR) fused aerial imagery**, using a **MobileViTv2-100 Early Fusion** model trained end-to-end with Focal + Dice loss.

---

## Architecture

| Component | Detail |
|---|---|
| Backbone | MobileViTv2-100 (pretrained, ImageNet) |
| Fusion strategy | **Early Fusion** — 4-channel RGBT input projected to 3-ch via learnable 1×1 conv |
| Decoder | 5-stage bilinear upsampling head (32× total) |
| Loss | Focal Loss (γ=2, weight=0.4) + Soft Dice (weight=0.6) |
| LR schedule | Linear warmup → Cosine annealing |
| Input size | 256 × 256 (fused TIFF) |
| Output | Per-pixel class mask `[B, 5, H, W]` |

### Class Map

| ID | Class | Notes |
|---|---|---|
| 0 | Background | Dominant class; down-weighted by Focal Loss |
| 1 | Crack | Surface fractures |
| 2 | Spall | Concrete spalling / delamination |
| 3 | Delamination | Reserved (no annotations in current dataset) |
| 4 | Moisture | Moisture infiltration; inverse-frequency boosted ×1.8 |

---

## Training Results (EarlyFusionPipelineV2 — 200 epochs)

### Best Validation Metrics (Epoch 200)

| Metric | Value |
|---|---|
| Loss | 0.4110 |
| Pixel Accuracy | 71.89% |
| Mean IoU | 47.12% |
| Mean Dice | 58.95% |
| Macro F1 | 52.82% |
| mAP | 51.59% |

### Per-Class Breakdown

| Class | Accuracy | IoU | Dice | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Background | 0.7295 | 0.7158 | 0.8344 | 0.9950 | 0.7184 | 0.8344 |
| Crack | 0.9858 | 0.4467 | 0.6175 | 0.5491 | 0.7054 | 0.6175 |
| Spall | 0.9901 | 0.5955 | 0.7465 | 0.7552 | 0.7379 | 0.7465 |
| Delamination | 0.9924 | 0.5978 | 0.7483 | 0.7588 | 0.7380 | 0.7483 |
| Moisture | 0.7401 | 0.0003 | 0.0006 | 0.0003 | 0.8696 | 0.0006 |

> **Note on Moisture:** Near-zero IoU/F1 despite high recall (0.87) indicates the model predicts moisture almost everywhere — a class imbalance problem. High recall but extremely low precision (0.03%) means false positives dominate. Addressed in future work.

Best checkpoint: `checkpoints/best_early_fusion_v2_epoch199.pth`

---

## Project Structure

```
THESIS/
├── CNNVIT.py               # Base model, trainer, metrics (DefectDetectionTrainer, DefectDetectionMetrics)
├── CNNVIT_v2.py            # EarlyFusionPipelineV2 — main training pipeline
├── late_fusion_model.py    # Late Fusion variant (separate RGB + IR streams)
├── channelstack.py         # Fuses RGB + IR TIFFs into 4-channel files
├── dataset.py              # COCOSegmentationDataset — data loading
├── dataset_utils.py        # CVAT XML → COCO JSON converter, dataset tools
├── visualization.py        # Training curves, confusion matrix, segmentation overlays
├── dashboard.py            # Streamlit inference dashboard
├── prepare_dataset.py      # Dataset preparation script
├── config.yaml             # All training hyperparameters
├── checkpoints/            # Saved model weights + training_history.json
├── visualizations/         # Output plots (training_history.png, confusion_matrix.png)
└── dataset/                # COCO-formatted dataset
    ├── instances_train.json
    └── instances_val.json
```

---

## Installation

**Requirements:** Python 3.8+, CUDA 11.0+ (recommended)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install timm pycocotools opencv-python numpy tqdm pyyaml \
            matplotlib seaborn scikit-learn streamlit
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Dataset Setup

### 1 — Fuse RGB + IR into 4-channel TIFFs

```bash
python channelstack.py
```

This writes `.tiff` files containing [R, G, B, IR] channels into the configured fused output directory.

### 2 — Convert CVAT XML annotations to COCO JSON

This runs automatically on first training via `CVATXMLToCOCOConverter` inside `CNNVIT_v2.py`, or manually:

```python
from dataset_utils import CVATXMLToCOCOConverter

converter = CVATXMLToCOCOConverter("path/to/annotations.xml")
converter.convert(
    output_train="instances_train.json",
    output_val="instances_val.json",
    train_ratio=0.8
)
```

### 3 — Expected Dataset Layout

```
dataset/
├── fused/
│   ├── fuse_Binondo/   ← 4-channel .tiff files
│   ├── FUSED/
│   └── ...
├── instances_train.json
└── instances_val.json
```

---

## Configuration (`config.yaml`)

```yaml
dataset:
  fused_dir:             "c:/Users/David/Documents/DLSU/THESIS/dataset/fused"
  annotation_dir:        "c:/Users/David/Documents/DLSU/THESIS/dataset"
  train_annotation_file: "instances_train.json"
  val_annotation_file:   "instances_val.json"
  xml_annotation:        "path/to/annotations.xml"
  num_classes: 5
  image_size:  256
  train_ratio: 0.8

training:
  batch_size:               4
  num_epochs:               200
  learning_rate:            0.001
  weight_decay:             0.0001
  early_stopping_patience:  20

checkpoint:
  save_dir: "./checkpoints"
```

---

## Training

```bash
python CNNVIT_v2.py
```

The pipeline will:
1. Convert CVAT XML → COCO JSON (skipped if already exists)
2. Compute data-driven class weights from annotation pixel counts
3. Train with Focal + Dice loss and cosine LR schedule
4. Save the best checkpoint to `checkpoints/best_early_fusion_v2_epochXXX.pth`
5. Save the full training history to `checkpoints/training_history.json`

To train programmatically:

```python
from CNNVIT_v2 import EarlyFusionPipelineV2

pipeline = EarlyFusionPipelineV2(
    fused_dir="./dataset/fused",
    annotation_dir="./dataset",
    num_classes=5,
    batch_size=4,
    num_epochs=200,
    image_size=256,
    checkpoint_dir="./checkpoints",
    lr=1e-3,
)

history = pipeline.train(
    train_annotation="instances_train.json",
    val_annotation="instances_val.json",
    early_stop_patience=20,
)
```

---

## Visualization

After training, generate plots from the saved history file:

```bash
python visualization.py
# or specify a custom path:
python visualization.py ./checkpoints/training_history.json
```

This produces two files in `./visualizations/`:

| File | Contents |
|---|---|
| `training_history.png` | 6-panel grid: Loss, Accuracy, IoU, Dice, F1, mAP — Train vs Val per epoch, best epoch annotated |
| `confusion_matrix.png` | Pixel-count heatmap (normalized by row) + per-class Precision/Recall/F1 bar chart |

To call from code:

```python
from visualization import DefectVisualization
import json, numpy as np

with open("checkpoints/training_history.json") as f:
    history = json.load(f)

best_ep = int(np.argmax([h["mean_iou"] for h in history["val"]]))
cm = np.array(history["val"][best_ep]["confusion_matrix"])

viz = DefectVisualization(output_dir="./visualizations")
viz.plot_training_history(history)
viz.plot_confusion_matrix(cm, class_names=["Background","Crack","Spall","Delamination","Moisture"])
```

---

## Inference

### Single fused TIFF

```python
from CNNVIT_v2 import EarlyFusionPipelineV2

pipeline = EarlyFusionPipelineV2(
    fused_dir="./dataset/fused",
    annotation_dir="./dataset",
    num_classes=5,
)

result = pipeline.predict(
    fused_path="./dataset/fused/fuse_Binondo/000001.tiff",
    checkpoint_path="./checkpoints/best_early_fusion_v2_epoch199.pth"
)

# result["mask"]  — H×W numpy array of class IDs (0–4)
# result["rgb"]   — H×W×3 RGB image
```

### Streamlit Dashboard

```bash
streamlit run dashboard.py
```

---

## Data Augmentation

Applied to the training split only:

| Type | Transform |
|---|---|
| Geometric | Random horizontal/vertical flip, rotation ±15°, affine |
| Color | Brightness, contrast, saturation, hue jitter |
| Blur | Gaussian blur (random kernel) |
| Noise | Gaussian noise injection |

---

## Troubleshooting

**Out of memory (OOM)**
- Reduce `batch_size` (current: 4) or `image_size` (current: 256)

**Moisture IoU near zero**
- The class is severely imbalanced. The current ×1.8 weight boost is insufficient. Consider oversampling moisture images, using a higher boost factor, or a threshold-based post-processing step.

**COCO JSON missing / empty**
- Re-run `CVATXMLToCOCOConverter.convert()` and verify the XML contains `<polygon>` annotations for all defect classes.

**Slow training (~2 min/epoch on CPU)**
- Confirm CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce `num_workers` in `DataLoader` if encountering deadlocks on Windows.

---

## References

- Mehta & Rastegari — [MobileViT (2021)](https://arxiv.org/abs/2110.02178)
- Mehta & Rastegari — [Separable Self-Attention for MobileViTv2 (2022)](https://arxiv.org/abs/2206.02680)
- Lin et al. — [Focal Loss (RetinaNet, 2017)](https://arxiv.org/abs/1708.02002)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [timm — PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)
