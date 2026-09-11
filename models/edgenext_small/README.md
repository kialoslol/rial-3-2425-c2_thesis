# edgenext_small — EdgeNeXt-Small

Early-fusion pattern identical to `models/mobilenet_v4` (learnable 1x1 conv
projects the 4-channel fused RGBT input down to 3 channels so a pretrained
3-channel ImageNet backbone can be reused unmodified), swapping in an
EdgeNeXt-Small backbone from `timm`.

## Architecture

| Component | Detail |
|---|---|
| Backbone | `edgenext_small.usi_in1k` (`timm`, ImageNet-pretrained), falls back to `edgenext_x_small.in1k` |
| Fusion | Learnable 1x1 conv, 4ch -> 3ch (identity-initialised on RGB, zero on IR) — same scheme as `mobilenet_v4`/`early_fusion_v1`/`v2` |
| Decoder | `common.heads.MobileViTSegmentationHead` — 5x 2x bilinear upsample stages |
| Loss | `common.losses.FocalDiceLoss` |
| Input | `[B, 4, H, W]` (fused RGB+IR TIFF), trained at the shared `config.yaml` resolution (384) |
| Output | `[B, 5, H, W]` logits |

## Files

- `model.py` — `EdgeNeXtSegmentationModel`
- `train.py` — `EdgeNeXtPipeline` (train / predict) + config-driven `__main__`

## Status

New model, not yet trained. Verified end-to-end: pretrained weights
download/load correctly, forward + backward pass at the real config image
size (384) reaches every trainable parameter (including `channel_proj`).

Uses the same training recipe as `mobilenet_v4`/`early_fusion_v2`/
`CNN_VIT_SELF` (data-driven class weights, moisture oversampling, cosine
annealing with warm restarts, partial-mIoU early stopping), so any accuracy
difference vs. those models reflects the backbone, not the training setup.

## Usage

```bash
python models/edgenext_small/model.py   # shape smoke-test (no training)
python models/edgenext_small/train.py   # full training run, reads config.yaml
```

Checkpoints save to `models/edgenext_small/checkpoints/` as
`best_edgenext_epoch{N:03d}.pth`.
