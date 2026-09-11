# deit_small — DeiT-Small/patch16/224 (pure-ViT baseline)

Early-fusion pattern identical to `models/mobilenet_v4` (learnable 1x1 conv
projects the 4-channel fused RGBT input down to 3 channels so a pretrained
3-channel ImageNet backbone can be reused unmodified), swapping in a plain
DeiT-Small backbone from `timm` as a pure-ViT baseline (no convolution,
only patch embedding + self-attention).

Requested as `deit_small_patch16_224` — resolved to the actual pretrained
tag `deit_small_patch16_224.fb_in1k` (the bare name has no pretrained
weights in this timm version; verified via `timm.list_models(pretrained=True)`).

## Decoder exception — read this first

Every other model in `models/` uses a stride-32 backbone: at the shared
384px input this yields a 12x12 feature grid, which `common.heads.
MobileViTSegmentationHead`'s fixed 5-stage/32x-upsample decoder maps back to
384x384. **DeiT-Small uses 16x16 patches (stride 16)**, so at 384px it
produces a 24x24 grid — running that through the 32x head would output
768x768, mismatching the 384x384 ground-truth mask.

This model therefore defines its own `DeiTSegmentationHead` in `model.py`
(NOT added to `common/heads.py`): the same conv+BN+ReLU+upsample block
design, with 4 stages (16x total) instead of 5, correctly matching this
backbone's native stride. This is the only architectural deviation from the
shared pattern, and it's scoped entirely to this file.

Two more DeiT-specific details (also documented in `model.py`):
- `dynamic_img_size=True` is passed to `timm.create_model` so the backbone
  accepts 384px input instead of hard-asserting on its pretrained 224px
  config (verified: position embeddings interpolate correctly).
- `forward_features()` returns tokens with a prepended CLS token
  (`[B, 1+N, C]`); feature extraction slices it off via
  `backbone.num_prefix_tokens` before reshaping to a spatial grid, since the
  plain `sqrt(N)` reshape used by every other model here would break on
  DeiT's 577 = 1 + 24×24 tokens.

## Architecture

| Component | Detail |
|---|---|
| Backbone | `deit_small_patch16_224.fb_in1k` (`timm`, ImageNet-pretrained, `dynamic_img_size=True`), falls back to `deit_tiny_patch16_224.fb_in1k` |
| Fusion | Learnable 1x1 conv, 4ch -> 3ch (identity-initialised on RGB, zero on IR) — same scheme as `mobilenet_v4`/`early_fusion_v1`/`v2` |
| Decoder | `DeiTSegmentationHead` (local to this file) — 4x 2x bilinear upsample stages (16x total) |
| Loss | `common.losses.FocalDiceLoss` |
| Input | `[B, 4, H, W]` (fused RGB+IR TIFF), trained at the shared `config.yaml` resolution (384) |
| Output | `[B, 5, H, W]` logits |

## Files

- `model.py` — `DeiTSegmentationModel`, `DeiTSegmentationHead`
- `train.py` — `DeiTPipeline` (train / predict) + config-driven `__main__`

## Status

New model, not yet trained. Verified end-to-end: pretrained weights
download/load correctly with `dynamic_img_size=True`, forward + backward
pass at the real config image size (384) reaches every trainable parameter
(including `channel_proj`), and output shape matches the 384x384 mask.

Uses the same training recipe as `mobilenet_v4`/`early_fusion_v2`/
`CNN_VIT_SELF` (data-driven class weights, moisture oversampling, cosine
annealing with warm restarts, partial-mIoU early stopping), so any accuracy
difference vs. those models reflects the backbone, not the training setup.

## Usage

```bash
python models/deit_small/model.py   # shape smoke-test (no training)
python models/deit_small/train.py   # full training run, reads config.yaml
```

Checkpoints save to `models/deit_small/checkpoints/` as
`best_deit_epoch{N:03d}.pth`.
