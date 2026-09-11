# Project Setup & Installation Guide

## Project Structure

```
THESIS/
├── common/                     # Shared infrastructure used by every model
│   ├── heads.py                #   MobileViTSegmentationHead (bilinear decoder head)
│   ├── trainer.py              #   DefectDetectionTrainer, DefectDetectionMetrics, checkpoint helpers
│   ├── losses.py                #   FocalDiceLoss (shared by early_fusion_v2, CNN_VIT_SELF, mobilenet_v4)
│   ├── dataset.py               #   COCOSegmentationDataset, MultiModalDataset
│   ├── dataset_utils.py        #   CVAT/Pascal-VOC XML -> COCO JSON converters, class-weight helpers
│   └── visualization.py        #   Training curve / confusion matrix plotting
│
├── models/                     # One folder per model/variation — see each README.md
│   ├── single_modal_v1/        #   MobileViT-S, single 3-channel image
│   ├── single_modal_v2/        #   MobileViTv2-100, single 3-channel image
│   ├── early_fusion_v1/        #   MobileViT-S, 4-channel RGBT projected to 3ch
│   ├── early_fusion_v2/        #   MobileViTv2-100 + ASPP decoder + Focal/Dice loss (primary model)
│   ├── CNN_VIT_SELF/            #   From-scratch CNN + self-attention hybrid (no pretrained backbone)
│   ├── mobilenet_v4/            #   MobileNetV4-Conv-Medium backbone
│   ├── late_fusion_v1/         #   Dual MobileViT-S backbones (RGB + IRT), decoder-level attention fusion
│   ├── late_fusion_v2/         #   Dual MobileViTv2-100 backbones, decoder-level attention fusion
│   ├── yolov11_defect_detection/ #  YOLOv11 single-class object detector (bounding boxes, not segmentation)
│   ├── fastvit_sa12/           #   FastViT-SA12 (trains unfused, reparameterize() on export only)
│   ├── efficientformerv2_s2/   #   EfficientFormerV2-S2 (native 224px — resolution exception, see its README)
│   ├── edgenext_small/         #   EdgeNeXt-Small
│   ├── swiftformer_l1/         #   SwiftFormer-L1
│   ├── repvit_m1_1/            #   RepViT-M1.1 (trains unfused, reparameterize() on export only)
│   ├── efficientnet_b0/        #   EfficientNet-B0 — pure-CNN baseline
│   ├── deit_small/             #   DeiT-Small/patch16/224 — pure-ViT baseline (custom 4-stage decoder)
│   └── <name>/
│       ├── model.py            #   nn.Module definition
│       ├── train.py            #   Pipeline class + config-driven __main__
│       ├── checkpoints/        #   Saved .pth files for this model only
│       └── README.md           #   Architecture, usage, and status for this model
│
├── scripts/                    # Repo-wide utility scripts (not tied to one model)
│   ├── channelstack.py         #   Fuse RGB + IR into 4-channel TIFFs
│   ├── prepare_dataset.py      #   Legacy Pascal-VOC dataset preparation
│   ├── quickstart.py           #   Legacy end-to-end demo (predates CVAT/config.yaml flow)
│   ├── test_installation.py    #   Verify environment + imports
│   ├── count_classes.py        #   Count annotation instances per class in a CVAT XML
│   └── rename_files.py         #   Batch-rename files in a folder
│
├── docs/                       # Reference documents (not code)
│   └── MobileRankings_TIMM.docx  # TIMM mobile-model ranking behind the mobilenet_v4 backbone choice
│
├── dashboard.py                 # Streamlit evaluation UI (loads any model's checkpoints)
├── config.yaml                  # Shared dataset/training hyperparameters (per-model checkpoint dirs override save_dir)
├── requirements.txt
│
├── README.md                    # Full documentation and model overview
├── DOCUMENTATION.txt            # Deep technical reference (architecture, training pipeline internals)
├── DATASET_FORMAT_GUIDE.md      # Dataset format specifications
├── SETUP.md                     # This file
│
├── RGB_FITTED/ IRT_FITTED/ FUSED/ ANNOTATED_DATA/  # Shared source imagery (RGB, IR, fused, CVAT XML)
└── dataset/                     # COCO-formatted dataset (shared across models)
    ├── instances_train.json
    └── instances_val.json
```

Each model folder is self-contained for training/inference purposes (its own
`model.py`, `train.py`, `checkpoints/`), while importing shared, non-model-specific
code from `common/`. This means a bug fix to the trainer or dataset loader in
`common/` applies to every model at once, but each model's checkpoints never
collide with another's.

## Installation

### 1. Clone/Download Project
```bash
cd c:\Users\David\Documents\DLSU\THESIS
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/Scripts/activate  # Windows
# or: source venv/bin/activate  # Linux/Mac

# OR using conda
conda create -n defect_detection python=3.10
conda activate defect_detection
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python scripts/test_installation.py
```

This checks package imports, GPU availability, `config.yaml`, dataset
directories, and a model instantiation + forward pass.

## Quick Start

Every model can be trained directly from the repo root — each `train.py`
resolves `config.yaml` and its own `checkpoints/` folder relative to its own
file location, so the current working directory doesn't matter:

```bash
python models/early_fusion_v2/train.py     # primary model — see README.md
python models/late_fusion_v1/train.py
python models/single_modal_v1/train.py
# ...one train.py per model folder under models/
```

See the top-level `README.md` for a full model comparison table, and each
`models/<name>/README.md` for that model's architecture and any known
limitations.

### Running the dashboard

```bash
streamlit run dashboard.py
```

The dashboard scans `models/*/checkpoints/*.pth` and picks the right
architecture automatically from the checkpoint filename prefix.

## GPU Acceleration

### Check GPU Availability
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

### Force CPU
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'common'` or `'models'`
**Fix**: Run scripts from the repo root (`python models/<name>/train.py`,
not `cd models/<name> && python train.py`). Every entry-point script inserts
the repo root onto `sys.path` itself, so this is the only requirement.

### Issue: SSL error / `LocalEntryNotFoundError` when downloading pretrained weights
**Fix**: Some networks (university/corporate proxies, some antivirus
software) TLS-intercept HTTPS with their own certificate — Windows and
`curl` trust it, but Python's `certifi` bundle doesn't, so `timm`'s Hugging
Face Hub download fails with an SSL certificate error. `common/__init__.py`
already calls `truststore.inject_into_ssl()` to point Python's SSL
verification at the OS trust store instead — just make sure `truststore` is
installed (it's in `requirements.txt`).

### Issue: COCO annotation not found
**Fix**: Each `models/<name>/train.py` regenerates
`dataset/instances_train.json` / `instances_val.json` from
`ANNOTATED_DATA/*.xml` automatically on first run via
`common.dataset_utils.CVATXMLToCOCOConverter`, driven by `config.yaml`'s
`dataset.xml_annotation` key.

### Issue: Out of memory (OOM)
**Fix**: Reduce `training.batch_size` or `dataset.image_size` in `config.yaml`.

### Issue: Dataset loading is slow
**Fix**:
- Reduce `num_workers` in the `DataLoader` calls inside `train.py` if you hit
  deadlocks on Windows
- Use SSD storage if available

## Model Metrics

`common/trainer.py`'s `DefectDetectionMetrics` computes and tracks:
- **Accuracy**: Pixel-level classification accuracy
- **IoU**: Intersection over Union (per-class and mean)
- **Dice**: Dice coefficient (per-class and mean)
- **F1-Score / mAP**: Precision/recall-derived metrics over defect classes (background excluded)

## Citation

If you use this model in your thesis:
```bibtex
@article{mehta2021mobilevit,
  title={MobileViT: Light-weight Vision Transformers},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2110.02178},
  year={2021}
}
```

## Support & Debugging

1. Check `config.yaml` for correct dataset paths
2. Verify the COCO JSON format matches `DATASET_FORMAT_GUIDE.md`
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Run `python scripts/test_installation.py` for a full environment check

For detailed dataset format information, see `DATASET_FORMAT_GUIDE.md`.
For per-model architecture details, see `models/<name>/README.md`.
