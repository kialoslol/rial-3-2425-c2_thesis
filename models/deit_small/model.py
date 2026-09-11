"""
deit_small — DeiT-Small/patch16/224 backbone (timm, pretrained). Pure-ViT
baseline.

Same early-fusion pattern as models/mobilenet_v4: a learnable 1x1 conv
projects the 4-channel fused RGBT input down to 3 channels (identity-
initialised on RGB, zero on IR) so the pretrained 3-channel ImageNet
backbone can be reused unmodified.

DECODER EXCEPTION — read before assuming this uses MobileViTSegmentationHead:
Every other model in models/ uses a stride-32 backbone (384px input -> 12x12
feature grid), matching common/heads.py's MobileViTSegmentationHead, which
always applies exactly 5 stages of 2x upsampling (32x total) to land back on
the input resolution. DeiT-Small uses 16x16 patches (stride 16), so at 384px
it produces a 24x24 grid; 5x 2x-upsample from a 24x24 grid would land on
768x768, not 384x384, mismatching the ground-truth mask shape. This model
therefore uses a local `DeiTSegmentationHead` (defined below, NOT added to
common/heads.py) with 4 upsample stages (16x total), correctly matching this
backbone's native stride. Same conv+BN+ReLU+upsample block design as
MobileViTSegmentationHead, just one stage shorter.

Two more DeiT-specific details handled here:
  - `dynamic_img_size=True` is passed to timm.create_model so the backbone
    accepts 384px input instead of hard-asserting on its pretrained 224px
    config (verified: interpolates the position embedding correctly).
  - forward_features() returns a token sequence with a prepended CLS token
    ([B, 1+N, C]); `_extract_features` slices it off via
    `backbone.num_prefix_tokens` before reshaping to a spatial grid — the
    plain sqrt(N) reshape used by every other model here would break on
    DeiT's 577 = 1 + 24*24 tokens.

Class IDs (shared across all models):
  0  background
  1  crack
  2  spall
  3  delamination  (reserved; no annotations yet)
  4  moisture
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.losses import FocalDiceLoss

logger = logging.getLogger(__name__)

BACKBONE_NAME = "deit_small_patch16_224.fb_in1k"
BACKBONE_FALLBACK = "deit_tiny_patch16_224.fb_in1k"


class DeiTSegmentationHead(nn.Module):
    """
    4 x 2x bilinear upsample stages -> 16x total upsampling.

    Matches DeiT-Small/patch16's native stride (16), unlike the shared
    common.heads.MobileViTSegmentationHead (32x, for stride-32 backbones).
    See the module docstring for why this can't just reuse that head.
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        d = hidden_dim
        d2 = hidden_dim // 2
        d4 = hidden_dim // 4

        def _block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            )

        self.decoder = nn.Sequential(
            _block(in_channels, d),   # e.g. 24 ->  48
            _block(d,           d2),  #      48 ->  96
            _block(d2,          d4),  #      96 -> 192
            _block(d4,          d4),  #     192 -> 384
        )
        self.seg_head = nn.Conv2d(d4, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg_head(self.decoder(x))


class DeiTSegmentationModel(nn.Module):
    """
    Early Fusion segmentation model using DeiT-Small/patch16/224 (pure-ViT
    baseline — no convolution, only attention token-mixing).

    Input tensor shape : [B, 4, H, W]   (fused RGB+IR)
    Output tensor shape: [B, num_classes, H, W]
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        hidden_dim: int = 256,
        in_channels: int = 4,
        class_weights: list | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Channel projection: [B, in_channels, H, W] -> [B, 3, H, W]
        # Identity-initialised on RGB, zero on IR — same scheme as
        # models/early_fusion_v1, models/early_fusion_v2, models/mobilenet_v4.
        self.channel_proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        nn.init.zeros_(self.channel_proj.weight)
        for i in range(3):
            self.channel_proj.weight.data[i, i, 0, 0] = 1.0

        try:
            self.backbone = timm.create_model(
                BACKBONE_NAME, pretrained=pretrained, dynamic_img_size=True
            )
            logger.info("Backbone loaded: %s", BACKBONE_NAME)
        except Exception:
            logger.warning("%s unavailable — falling back to %s", BACKBONE_NAME, BACKBONE_FALLBACK)
            self.backbone = timm.create_model(
                BACKBONE_FALLBACK, pretrained=pretrained, dynamic_img_size=True
            )

        backbone_ch = getattr(self.backbone, "num_features", 384)
        self.seg_head = DeiTSegmentationHead(backbone_ch, num_classes, hidden_dim)

        if class_weights is None:
            class_weights = [0.5, 3.0, 2.0, 1.0, 10.0]
        weights = torch.tensor(class_weights[:num_classes], dtype=torch.float32)
        self.criterion = FocalDiceLoss(weights, num_classes)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        if feats.dim() == 4:
            return feats
        if feats.dim() == 3:
            # Strip CLS/register prefix tokens before reshaping to a grid —
            # DeiT prepends 1 CLS token, so plain sqrt(N) would break.
            num_prefix = getattr(self.backbone, "num_prefix_tokens", 0) or 0
            feats = feats[:, num_prefix:, :]
            B, N, C = feats.shape
            h = w = int(N ** 0.5)
            return feats.permute(0, 2, 1).reshape(B, C, h, w)
        raise ValueError(f"Unexpected feature dim: {feats.dim()}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg_head(self._extract_features(self.channel_proj(x)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = DeiTSegmentationModel(num_classes=5, pretrained=False)
    model.eval()

    B, H = 2, 256
    x = torch.randn(B, 4, H, H)
    with torch.no_grad():
        out = model(x)

    expected = (B, 5, H, H)
    status = "PASS" if out.shape == expected else "FAIL"
    print(f"{status}  output shape: {tuple(out.shape)}  expected: {expected}")
