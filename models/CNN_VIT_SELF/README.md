# CNN-ViT-Self — from-scratch CNN + self-attention hybrid

Every other model in this repo builds on a `timm` **pretrained** MobileViT or
MobileViTv2 backbone. This one doesn't — the convolutional stem, CNN stages,
and self-attention blocks are all defined and trained **from scratch** here,
with no ImageNet-pretrained weights anywhere in the network.

## Architecture

```
Fused RGBT [B, 4, H, W]
    -> Conv stem                         (4   -> 32,  H    -> H/2)
    -> CNNStage    (local conv only)     (32  -> 64,  H/2  -> H/4)
    -> CNNStage    (local conv only)     (64  -> 128, H/4  -> H/8)
    -> CNNViTBlock (local conv + attn)   (128 -> 256, H/8  -> H/16)
    -> CNNViTBlock (local conv + attn)   (256 -> 384, H/16 -> H/32)
    -> common.heads.MobileViTSegmentationHead (5x 2x upsample -> H)
    -> logits [B, num_classes, H, W]
```

| Component | Detail |
|---|---|
| Stem + early stages | Plain depthwise-separable ("inverted residual") conv blocks — no attention, since self-attention over a high-resolution token grid is expensive |
| `CNNViTBlock` (stages 3-4) | Local depthwise-separable conv (handles the 2x downsample) + one pre-norm Transformer encoder layer (multi-head self-attention + MLP) over the flattened feature map, fused back via a 1x1 conv + residual |
| Positional encoding | 2D sinusoidal, computed fresh per forward pass from the actual feature-map size — not a fixed-size learned table, so the model isn't locked to one `image_size` |
| Decoder | `common.heads.MobileViTSegmentationHead` (same shared decoder as the single-modal / early-fusion-v1 models) |
| Loss | `common.losses.FocalDiceLoss` (same loss as `early_fusion_v2`) |
| Input | `[B, 4, H, W]` (fused RGB+IR TIFF) — consumed **directly**, no channel-projection layer |
| Output | `[B, 5, H, W]` logits |
| Parameters | ~3.55M (no pretrained backbone to carry) |

### Why no channel-projection layer

`early_fusion_v1`/`early_fusion_v2` project the 4-channel fused input down to
3 channels with a learnable 1x1 conv, specifically so the result fits a
pretrained 3-channel ImageNet backbone. This model has no pretrained weights
to protect, so its very first convolution (`stem`) just accepts all 4
channels natively.

### Why attention only in the last two stages

Mirrors the core idea MobileViT itself is built on — local features first
(cheap convolutions at high resolution), global reasoning once the grid is
small enough that full self-attention is affordable — just reimplemented
here with plain `nn.MultiheadAttention` instead of a pretrained ViT.

## Files

- `model.py` — `CNNViTSelfSegmentationModel`, `CNNStage`, `CNNViTBlock`, 2D sinusoidal positional encoding
- `train.py` — `CNNViTSelfPipeline` (train / predict) + config-driven `__main__`

## Status

New model, not yet trained to convergence. Verified end-to-end: forward +
backward pass (gradients reach every parameter), and one full real training
epoch against the actual dataset (`ANNOTATED_DATA/`) completed cleanly —
loss decreases, checkpointing/pruning works, metrics compute correctly. No
multi-epoch run or accuracy comparison against the other models has been
done yet.

Uses the same training recipe as `early_fusion_v2` (data-driven class
weights, moisture oversampling, cosine annealing with warm restarts, partial
mIoU early stopping) so any accuracy difference reflects the backbone, not
the training setup.

## Usage

```bash
python models/CNN_VIT_SELF/model.py   # shape smoke-test (no training)
python models/CNN_VIT_SELF/train.py   # full training run, reads config.yaml
```

Checkpoints save to `models/CNN_VIT_SELF/checkpoints/` as
`best_cnn_vit_self_epoch{N:03d}.pth`, and the trained model is automatically
picked up by `dashboard.py` (routed by the `cnn_vit_self` filename prefix).
