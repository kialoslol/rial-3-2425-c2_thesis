"""
Early Fusion model (v2) — MobileViTv2-100 backbone + ASPP decoder + Focal/Dice loss.

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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.losses import FocalDiceLoss

logger = logging.getLogger(__name__)

# Backward-compatible alias (old module name was private, and this class used
# to be defined here before it moved to common/losses.py for reuse)
_FocalDiceLoss = FocalDiceLoss


# ---------------------------------------------------------------------------
# ASPP decoder
# ---------------------------------------------------------------------------

class ASPPDecoder(nn.Module):
    """
    Atrous Spatial Pyramid Pooling decoder.

    Replaces the plain bilinear upsampling head to recover fine structural
    detail (crack edges, moisture boundaries) lost in deep backbone features.

    Forward signature: forward(feat, target_size) where target_size = (H, W)
    of the original input image so the output is always aligned to it.
    """

    def __init__(self, in_channels: int, num_classes: int, aspp_channels: int = 256):
        super().__init__()

        def _branch(kernel, dilation):
            pad = dilation if kernel > 1 else 0
            return nn.Sequential(
                nn.Conv2d(in_channels, aspp_channels, kernel,
                          padding=pad, dilation=dilation, bias=False),
                nn.BatchNorm2d(aspp_channels),
                nn.ReLU(inplace=True),
            )

        self.b1x1  = _branch(1, 1)
        self.b_d6  = _branch(3, 6)
        self.b_d12 = _branch(3, 12)
        self.b_d18 = _branch(3, 18)
        self.gap   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, aspp_channels, 1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(aspp_channels * 5, aspp_channels, 1, bias=False),
            nn.BatchNorm2d(aspp_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        # Two progressive 2x upsample steps with conv refinement
        self.up1 = nn.Sequential(
            nn.Conv2d(aspp_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.seg_head = nn.Conv2d(64, num_classes, 1)

    def forward(self, feat: torch.Tensor, target_size: tuple) -> torch.Tensor:
        h, w  = feat.shape[2], feat.shape[3]
        gap   = F.interpolate(self.gap(feat), size=(h, w),
                              mode="bilinear", align_corners=False)
        cat   = torch.cat([self.b1x1(feat), self.b_d6(feat),
                           self.b_d12(feat), self.b_d18(feat), gap], dim=1)
        proj  = self.project(cat)
        proj  = self.up2(self.up1(proj))
        # Final bilinear upsample to match original input resolution
        proj  = F.interpolate(proj, size=target_size,
                              mode="bilinear", align_corners=False)
        return self.seg_head(proj)


# ---------------------------------------------------------------------------
# Early Fusion model  (4-channel RGBT -> MobileViTv2 + ASPP decoder)
# ---------------------------------------------------------------------------

class EarlyFusionSegmentationModelV2(nn.Module):
    """
    Early Fusion segmentation model using MobileViTv2-100 + ASPP decoder.

    A learnable 1x1 channel-projection layer maps the 4-channel fused TIFF
    (RGB + IR) down to 3 channels before the pretrained MobileViTv2 backbone.
    The ASPP decoder replaces plain bilinear upsampling to recover fine
    structural detail (crack edges, moisture boundaries).

    Input tensor shape : [B, 4, H, W]
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

        self.channel_proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        nn.init.zeros_(self.channel_proj.weight)
        for i in range(3):
            self.channel_proj.weight.data[i, i, 0, 0] = 1.0

        try:
            self.backbone = timm.create_model("mobilevitv2_100", pretrained=pretrained)
            logger.info("Backbone loaded: mobilevitv2_100 (early fusion + ASPP)")
        except Exception:
            logger.warning("mobilevitv2_100 unavailable — falling back to mobilevitv2_075")
            self.backbone = timm.create_model("mobilevitv2_075", pretrained=pretrained)

        backbone_ch = getattr(self.backbone, "num_features", 256)
        self.decoder = ASPPDecoder(backbone_ch, num_classes, aspp_channels=hidden_dim)

        if class_weights is None:
            class_weights = [0.2, 3.0, 2.0, 1.0, 3.0]
        weights = torch.tensor(class_weights[:num_classes], dtype=torch.float32)
        self.criterion = FocalDiceLoss(weights, num_classes)

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        if feats.dim() == 4:
            return feats
        if feats.dim() == 3:
            B, N, C = feats.shape
            h = w = int(N ** 0.5)
            return feats.permute(0, 2, 1).reshape(B, C, h, w)
        raise ValueError(f"Unexpected feature dim: {feats.dim()}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (x.shape[2], x.shape[3])
        feats = self._extract_features(self.channel_proj(x))
        return self.decoder(feats, target_size)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = EarlyFusionSegmentationModelV2(num_classes=5, pretrained=False)
    model.eval()

    B, H = 2, 256
    x = torch.randn(B, 4, H, H)
    with torch.no_grad():
        out = model(x)

    expected = (B, 5, H, H)
    status = "PASS" if out.shape == expected else "FAIL"
    print(f"{status}  output shape: {tuple(out.shape)}  expected: {expected}")
