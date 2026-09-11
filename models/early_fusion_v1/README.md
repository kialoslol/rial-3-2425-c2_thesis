# Early Fusion v1 — MobileViT-S

4-channel RGBT input projected to 3 channels via a learnable 1x1 convolution,
then fed through a MobileViT-S backbone. Superseded by `models/early_fusion_v2`
(ASPP decoder + Focal/Dice loss + moisture oversampling), kept here as the
earlier baseline.

| Component | Detail |
|---|---|
| Backbone | MobileViT-S (`timm`, ImageNet-pretrained), falls back to `mobilenetv3_small_100` |
| Fusion | Learnable 1x1 conv, 4ch -> 3ch (identity-initialised on RGB, zero on IR) |
| Decoder | `common.heads.MobileViTSegmentationHead` — 5x 2x bilinear upsample stages |
| Loss | Weighted `CrossEntropyLoss` (`[0.2, 3.0, 2.0, 1.0, 3.0]`) |
| Input | `[B, 4, H, W]` (fused RGB+IR TIFF) |
| Output | `[B, 5, H, W]` logits |

## Files

- `model.py` — `EarlyFusionSegmentationModel`
- `train.py` — `EarlyFusionPipeline` (train / predict) + config-driven `__main__`

## Usage

```bash
python models/early_fusion_v1/model.py   # shape smoke-test (no training)
python models/early_fusion_v1/train.py   # full training run, reads config.yaml
```

Reads 4-channel fused TIFFs from `config.yaml`'s `dataset.fused_dir`
(default `ANNOTATED_DATA/`), and the COCO annotation JSON from
`dataset.annotation_dir`. Checkpoints save to `models/early_fusion_v1/checkpoints/`
as `best_early_fusion_epoch{N:03d}.pth`.
