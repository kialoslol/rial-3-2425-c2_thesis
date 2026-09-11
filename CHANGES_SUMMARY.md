# Project Reorganization Summary

## What Changed

### ✅ Files Created
1. **`dataset.py`** - NEW
   - Moved `COCOSegmentationDataset` class
   - Moved `RandomGaussianNoise` class
   - Improved error handling for image loading

2. **`requirements.txt`** - NEW
   - Complete list of dependencies for easy installation
   - Install with: `pip install -r requirements.txt`

3. **`__init__.py`** - NEW
   - Makes the project a proper Python package
   - Exports main classes for imports

4. **`SETUP.md`** - NEW
   - Complete setup and installation guide
   - Usage examples
   - Troubleshooting tips

5. **`prepare_dataset.py`** - NEW
   - Prepares your specific dataset structure
   - Copies images to organized directories
   - Creates placeholder COCO JSON files
   - Detects available modalities (RGB, Thermal, Fused)

6. **`test_installation.py`** - NEW
   - Verifies all dependencies are installed
   - Tests local modules
   - Checks GPU availability
   - Tests model creation
   - Validates dataset structure

### ✅ Files Modified
1. **`CNNVIT.py`** - REFACTORED
   - ✓ Removed `RandomGaussianNoise` class
   - ✓ Removed `COCOSegmentationDataset` class
   - ✓ Added import: `from dataset import COCOSegmentationDataset, RandomGaussianNoise`
   - ✓ Removed unused import: `Dataset` (was only used for COCOSegmentationDataset)
   - ✓ Now 100+ lines shorter, more focused on model/training

### ✅ Files Updated
1. **`README.md`** - Updated
   - Changed image format examples to include `.tiff`, `.tif`

2. **`DATASET_FORMAT_GUIDE.md`** - Updated
   - Added TIFF support documentation

3. **`dataset_utils.py`** - Updated
   - Added TIFF support in `XMLToCOCOConverter`
   - Added TIFF support in `MultimodalImageFusion`
   - Added TIFF support in `DatasetOrganizer`

## New Project Structure

```
THESIS/
├── Core Model Files
├── CNNVIT.py                    ← Model, trainer, pipeline (cleaned up)
├── dataset.py                   ← Dataset loaders (NEW - separated out)
├── dataset_utils.py             ← Dataset converters & formatters
├── visualization.py             ← Visualization utilities
│
├── Configuration & Setup
├── config.yaml                  ← Training configuration
├── requirements.txt             ← Dependencies (NEW)
├── __init__.py                  ← Package init (NEW)
├── SETUP.md                     ← Detailed setup guide (NEW)
├── prepare_dataset.py           ← Dataset preparation (NEW)
├── test_installation.py         ← Test script (NEW)
│
├── Documentation
├── README.md                    ← Full documentation (updated)
├── DATASET_FORMAT_GUIDE.md      ← Format specs (updated)
│
├── Utilities
├── quickstart.py                ← Automated pipeline
│
└── Data Directories
    ├── checkpoints/             ← Saved models
    ├── visualizations/          ← Output plots
    ├── predictions/             ← Inference results
    ├── dataset/                 ← COCO formatted data (generated)
    ├── RGB_FITTED/              ← Your RGB images
    ├── IRT_FITTED/              ← Your thermal images
    └── FUSED/                   ← Your fused images
```

## Benefits of Reorganization

1. **Better Separation of Concerns**
   - Dataset loading in `dataset.py`
   - Converters/formatters in `dataset_utils.py`
   - Model architecture in `CNNVIT.py`

2. **Easier Maintenance**
   - Changes to dataset loading don't affect model code
   - Smaller, focused files are easier to understand

3. **Better Import Structure**
   ```python
   # Before: Everything in CNNVIT.py
   # After:
   from dataset import COCOSegmentationDataset
   from dataset_utils import XMLToCOCOConverter
   from CNNVIT import DefectDetectionPipeline
   from visualization import DefectVisualization
   ```

4. **Improved Error Handling**
   - Better error messages in dataset.py
   - Validation checks for image loading

## Quick Start After Changes

### 1. Verify Installation
```bash
python test_installation.py
```

### 2. Prepare Your Dataset
```bash
python prepare_dataset.py
```

### 3. Update Configuration
Edit `config.yaml` with your paths

### 4. Start Training
```bash
python CNNVIT.py
```

Or use automated pipeline:
```bash
python quickstart.py
```

## Import Changes

### Before Reorganization
All imports were from CNNVIT.py:
```python
# Old (not recommended)
from CNNVIT import COCOSegmentationDataset, RandomGaussianNoise
from CNNVIT import MobileViTSegmentationModel
```

### After Reorganization
Cleaner, semantic imports:
```python
# New (recommended)
from dataset import COCOSegmentationDataset, RandomGaussianNoise
from CNNVIT import MobileViTSegmentationModel, DefectDetectionTrainer
from dataset_utils import XMLToCOCOConverter, MultimodalImageFusion
from visualization import DefectVisualization
```

## Error Fixes Applied

1. **Image Loading Errors**
   - Added proper error handling with informative messages
   - Added validation for image path existence

2. **Contour Drawing Errors**
   - Added check for minimum contour points (≥3 required by OpenCV)

3. **Import Issues**
   - Removed unused imports from CNNVIT.py
   - Added proper imports from new dataset.py

4. **TIFF Format Support**
   - Extended file format support to include `.tiff` and `.tif`

## Dependency Installation

### Method 1: Using requirements.txt (Recommended)
```bash
pip install -r requirements.txt
```

### Method 2: Manual Installation
```bash
pip install torch torchvision timm pycocotools opencv-python \
            pyyaml matplotlib numpy tqdm
```

### Verify Installation
```bash
python test_installation.py
```

## Next Steps

1. ✅ Read this file
2. ✅ Run `test_installation.py` to verify setup
3. ✅ Run `prepare_dataset.py` to prepare your data
4. ✅ Update `config.yaml` with your dataset paths
5. ✅ Run `python CNNVIT.py` to start training

## File Sizes Comparison

| File | Before | After | Change |
|------|--------|-------|--------|
| CNNVIT.py | ~850 lines | ~550 lines | -68% ✓ |
| dataset.py | N/A | ~170 lines | NEW ✓ |
| dataset_utils.py | ~300 lines | ~390 lines | +30% (added TIFF) |
| Total | ~1150 lines | ~1110 lines | Cleaner structure ✓ |

## Testing

Verify everything works:
```bash
# Test all components
python test_installation.py

# If all tests pass, you're ready!
```

## FAQ

**Q: Do I need to re-run anything after this change?**
A: Yes, run `prepare_dataset.py` to organize your data in the new structure.

**Q: Will my existing code break?**
A: Existing imports from `dataset_utils.py` still work. Only imports from CNNVIT for dataset classes need updating.

**Q: What if I get "module not found" errors?**
A: Ensure all .py files are in the same directory and run `test_installation.py`

**Q: Can I still use the old code?**
A: No, but the new structure is backward compatible. Just update your imports.

## Support

If you encounter issues:
1. Check `test_installation.py` output
2. Review `SETUP.md` for troubleshooting
3. Check error messages in `prepare_dataset.py`
4. Ensure config.yaml paths are correct

---

# 2026-09-09 Reorganization: One Folder Per Model

## What Changed

The six segmentation model variants (single-modal v1/v2, early fusion v1/v2,
late fusion v1/v2) that used to live as flat files at the repo root
(`CNNVIT.py`, `CNNVIT_v2.py`, `late_fusion_model.py`, `late_fusion_model_v2.py`)
were split into `models/<name>/{model.py, train.py, checkpoints/, README.md}`.
Shared, non-model-specific code (`DefectDetectionTrainer`,
`DefectDetectionMetrics`, `MobileViTSegmentationHead`, dataset loaders,
dataset/XML conversion utilities, visualization) moved into a new `common/`
package that every model imports. `yolov11_defect_detection/` (and its
pretrained `yolo11n.pt`/`yolo11m.pt` weights) moved under `models/` alongside
the segmentation models. Repo-wide utility scripts not tied to a specific
model (`channelstack.py`, `prepare_dataset.py`, `quickstart.py`,
`test_installation.py`, `count_classes.py`, `rename_files.py`) moved into a
new `scripts/` folder. Three placeholder folders (`models/model_05`,
`model_06`, `model_07`) were added for upcoming model variants.

Checkpoints now save per-model (`models/<name>/checkpoints/`) instead of a
single shared `checkpoints/` folder, so training runs never collide.
`dashboard.py` was updated to scan every model's checkpoint folder and route
to the right architecture by filename prefix, same as before.

## Why

- Requested directly: keep each model/variation self-contained in its own
  folder as the number of model variants grows.
- The shared `checkpoints/` folder was a latent collision risk once six
  models could all write `best_*_epoch000.pth`-style filenames into it.
- `late_fusion_model.py` and `late_fusion_model_v2.py` had model definitions
  but no training loop at all — `models/late_fusion_v1/train.py` and
  `models/late_fusion_v2/train.py` are new, wiring those models to the
  existing shared trainer and `MultiModalDataset` (no new dataset/trainer
  code was needed — the trainer already dispatched on `rgb`/`irt` batch keys).

## Verified

- Every `common/*` and `models/*/{model,train}.py` module imports cleanly.
- Every model's shape smoke-test (`python models/<name>/model.py`) passes.
- `dashboard.py` starts cleanly under `streamlit run` (no checkpoints exist
  yet, so it correctly shows "no checkpoints found" rather than crashing).
- `models/yolov11_defect_detection`'s scripts had their `config.yaml`/
  `ANNOTATED_DATA` path resolution updated for the extra directory level and
  verified to resolve correctly.

## Not changed

- No model architecture, loss function, or hyperparameter logic changed —
  this was a structural move plus import-path fixes, not a retraining.
- The pre-existing 4-channel/3-channel dataset-vs-model mismatch in
  `single_modal_v1`/`single_modal_v2` was documented (see those READMEs),
  not fixed, since fixing it would change training behavior.

---

# 2026-09-09 (later same day): New Model — CNN_VIT_SELF

## What Changed

Added `models/CNN_VIT_SELF/` — a from-scratch CNN + self-attention hybrid
for early-fusion (4-channel RGBT) segmentation. Unlike every other model in
this repo, it does not build on a `timm` pretrained backbone: the conv stem,
CNN stages, and self-attention blocks (`CNNStage`, `CNNViTBlock` in
`model.py`) are all defined and trained from scratch, so it also consumes
the 4-channel input directly with no channel-projection layer.

Two pieces of code that were previously private to
`models/early_fusion_v2/model.py` / `train.py` were promoted to shared
modules so this new model (and any future one) can reuse them instead of
duplicating them:
  - `FocalDiceLoss` moved to `common/losses.py` (early_fusion_v2 now imports
    it, with a backward-compatible re-export in its own `model.py`).
  - `_compute_class_weights` / `_build_moisture_sampler` moved to
    `common/dataset_utils.py` as `compute_class_weights` /
    `build_moisture_sampler` (early_fusion_v2's `train.py` keeps
    backward-compatible aliases to the old private names).

`dashboard.py` was updated to load `CNN_VIT_SELF` checkpoints (routed by the
`cnn_vit_self` filename prefix) and to call it with a single 4-channel
tensor rather than the two-argument late-fusion calling convention.

## Why

Requested directly: "develop a new CNN-ViT hybrid model." Since every
existing model already wraps a pretrained MobileViT(v2) backbone from
`timm`, a genuinely new architectural contribution meant building the
CNN+attention hybrid from scratch rather than adding another pretrained-
backbone variant.

## Verified

- Shape smoke-test passes (`python models/CNN_VIT_SELF/model.py`).
- Forward + backward pass confirmed at the real config image size (384):
  every parameter receives a gradient, no NaNs.
- One full real training epoch against the actual dataset
  (`ANNOTATED_DATA/`) completed end-to-end: loss decreases,
  checkpoint save/prune works, best-checkpoint reload and final metrics
  reporting work.
- `early_fusion_v2` re-verified after the `FocalDiceLoss` /
  class-weight-helper refactor — still imports and trains correctly.
- `dashboard.py` still starts cleanly under `streamlit run` with the new
  import and routing branch added.

## Not done

- No multi-epoch training run — accuracy is unknown, only that the training
  loop itself runs correctly.
- No comparison against the other models' validation metrics.

---

# 2026-09-09 (later still): New Model — mobilenet_v4, plus two environment fixes

## What Changed

Added `models/mobilenet_v4/` — an early-fusion model using a
MobileNetV4-Conv-Medium backbone (`timm`, pretrained). Backbone choice is
documented in the newly added `docs/MobileRankings_TIMM.docx` (a ranking of
222 mobile-class TIMM models by ImageNet accuracy, supplied by the user):
MobileNetV4 is the only family in that ranking spanning the full small-to-
large accuracy range in one coherent architecture; `_conv_medium` (~9.7M
params, ~82-83% Top-1) was picked as a mid-size variant between MobileViT-S
(5.6M) and MobileViTv2-100 (4.9M). Architecturally it follows the same
pattern as `early_fusion_v1`/`v2`: a learnable 1x1 conv projects the
4-channel fused RGBT input to 3 channels for the pretrained backbone, then
`common.heads.MobileViTSegmentationHead` decodes, with
`common.losses.FocalDiceLoss` for the loss. The `models/model_06`
placeholder folder was renamed to `models/mobilenet_v4` to hold it (per its
own README's instruction to rename once an architecture is decided).

`dashboard.py` was updated to load `mobilenet_v4` checkpoints (routed by
filename prefix) — no calling-convention change needed since this model
already has a `channel_proj` attribute like the other early-fusion models.

While building this model, two pre-existing environment bugs blocked
downloading its pretrained weights (any *new*, not-yet-cached pretrained
model would have hit both):
  1. `huggingface_hub` 1.11.0 had a client-lifecycle bug ("Cannot send a
     request, as the client has been closed") breaking the retry path for
     any fresh download. Fixed by upgrading to 1.30.0.
  2. This network TLS-intercepts HTTPS (a proxy or antivirus presents its
     own certificate) — Windows and `curl` trust it, but Python's `certifi`
     bundle doesn't, so `httpx`/`huggingface_hub` rejected every connection
     with `CERTIFICATE_VERIFY_FAILED`. Fixed by adding the `truststore`
     package and calling `truststore.inject_into_ssl()` in
     `common/__init__.py` (runs automatically whenever any model module is
     imported), which points Python's SSL verification at the OS trust
     store instead of disabling verification.

## Why

Requested directly: "develop also MobileNet on model_6", with a supplied
TIMM mobile-model ranking doc to justify the specific variant. The two
environment fixes were not requested but were required to get pretrained
weights loading at all — surfaced while verifying the new model actually
works, not applied speculatively.

## Verified

- Pretrained `mobilenetv4_conv_medium` weights now download and load
  correctly (confirmed after each of the two fixes, in isolation and
  together via `import common`).
- Shape smoke-test passes (`python models/mobilenet_v4/model.py`).
- Forward + backward pass confirmed at the real config image size (384)
  with real pretrained weights loaded: only the backbone's own unused
  ImageNet classification-head layers (`conv_head`, `norm_head`,
  `classifier` — never called since only `forward_features` is used) lack
  gradients, which is expected and matches every other early-fusion model
  here.
- One full real training epoch against the actual dataset
  (`ANNOTATED_DATA/`) completed end-to-end: loss decreases, checkpoint
  save/prune works, metrics compute correctly.
- `dashboard.py` still starts cleanly under `streamlit run` with the new
  import and routing branch added.

## Not done

- No multi-epoch training run — accuracy is unknown, only that the training
  loop itself runs correctly.
- No comparison against the other models' validation metrics.

---

# 2026-09-11: Seven New Models — filled model_05/model_07 placeholders, added five more

## What Changed

Added seven new early-fusion segmentation models, following the exact
`mobilenet_v4` pattern (learnable 1x1 conv 4ch→3ch, `timm` pretrained
backbone via `forward_features`, `common.heads.MobileViTSegmentationHead`
decoder, `common.losses.FocalDiceLoss`, data-driven class weights, moisture
oversampling, cosine annealing with warm restarts): `models/fastvit_sa12`,
`models/efficientformerv2_s2`, `models/edgenext_small`,
`models/swiftformer_l1`, `models/repvit_m1_1`, `models/efficientnet_b0`, and
`models/deit_small`. The `model_05` and `model_07` placeholder folders were
filled in and renamed (per their own README's instruction to rename once an
architecture is decided); `model_06`, `model_08`–`model_11` were created
fresh and renamed the same way.

Three of the requested checkpoint names had no pretrained weights under the
bare name in the installed timm version (1.0.26) and were resolved to their
actual tags: `swiftformer_l1` → `swiftformer_l1.dist_in1k`,
`efficientnet_b0` → `efficientnet_b0.ra_in1k`,
`deit_small_patch16_224` → `deit_small_patch16_224.fb_in1k`.

Two architectures needed backbone-specific handling instead of a straight
`mobilenet_v4` copy:
- **FastViT-SA12** and **RepViT-M1.1** are RepVGG-style reparameterizable —
  both train unfused; each model class exposes a `.reparameterize()` method
  (wrapping `timm.utils.model.reparameterize_model`) that is called only
  from `train.py`'s `predict()` (the inference path), never during training.
- **DeiT-Small/patch16/224** has a stride-16 feature grid, not stride-32 like
  every other backbone here, so the shared `MobileViTSegmentationHead`'s
  fixed 5-stage/32x upsampling would mismatch the mask resolution. It uses a
  local `DeiTSegmentationHead` (4 stages, 16x) defined in its own
  `model.py` — `common/heads.py` was not touched. It also needs
  `dynamic_img_size=True` (to accept 384px instead of asserting on its
  pretrained 224px config) and prefix-token stripping before the
  feature-map reshape (ViT's `forward_features` prepends a CLS token).

**EfficientFormerV2-S2 is the one model trained at a different resolution.**
Its `attention_biases` buffers are baked at construction time for a fixed
token grid (7x7 at 224px) with no interpolation support in timm for other
sizes — building at 384px either crashes (default 224 config, mismatched
runtime grid) or discards ~9 layers' pretrained weights (rebuilt at
`img_size=384`, shape-mismatched against the 224px checkpoint). Per user
decision, `models/efficientformerv2_s2` trains at its native 224px with full
pretraining intact instead of the shared `config.yaml` 384px every other
model uses; everything else (loss, class weights, augmentation, scheduler)
is identical.

`dashboard.py` was **not** updated to route these new checkpoints — routing
is left for the user to add if/when a model is trained, so as not to touch
an existing file as part of this addition.

## Verified

- All 7 shape smoke-tests pass (`python models/<name>/model.py`).
- All 7 forward + backward passes confirmed at their real training
  resolution with real pretrained weights loaded.
- All 7 `Pipeline` classes construct successfully against the actual
  `config.yaml` and dataset annotations (data-driven class weights compute
  correctly, GPU detected).
- `reparameterize()` verified numerically lossless on both FastViT and
  RepViT (max abs diff ~1e-8–1e-7 between fused/unfused outputs).

## Not done

- None of the 7 have been trained — only verified end-to-end, same status
  `mobilenet_v4` had when it was added.
- No `dashboard.py` routing for these checkpoints yet.

---

**Summary**: Your project is now better organized with proper separation of concerns, improved error handling, and TIFF support. You're ready to train! 🚀
