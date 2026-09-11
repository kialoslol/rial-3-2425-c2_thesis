"""
Early Fusion model (v1) — 4-channel RGBT input projected to 3 channels,
then fed through a MobileViT-S backbone + bilinear-upsample decoder.

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
from common.heads import MobileViTSegmentationHead

logger = logging.getLogger(__name__)


class EarlyFusionSegmentationModel(nn.Module):
    """
    Early Fusion segmentation model.

    A learnable 1x1 channel-projection layer maps the 4-channel fused TIFF
    (RGB + IR) down to 3 channels, which are then fed into the pretrained
    MobileViT-S backbone. This keeps the backbone weights intact while
    allowing the network to learn an optimal combination of all four input
    channels from the start of training.

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

        # Channel projection: [B, in_channels, H, W] -> [B, 3, H, W]
        self.channel_proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        # Initialise: pass RGB channels through as identity; IR weight starts at 0
        nn.init.zeros_(self.channel_proj.weight)
        for i in range(3):
            self.channel_proj.weight.data[i, i, 0, 0] = 1.0

        try:
            self.backbone = timm.create_model("mobilevit_s", pretrained=pretrained)
            logger.info("Backbone loaded: mobilevit_s (early fusion)")
        except Exception:
            logger.warning("mobilevit_s unavailable — falling back to mobilenetv3_small_100")
            self.backbone = timm.create_model("mobilenetv3_small_100", pretrained=pretrained)

        backbone_ch = getattr(self.backbone, "num_features", 640)
        self.seg_head = MobileViTSegmentationHead(backbone_ch, num_classes, hidden_dim)

        if class_weights is None:
            class_weights = [0.2, 3.0, 2.0, 1.0, 3.0]
        weights = torch.tensor(class_weights[:num_classes], dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)

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
        return self.seg_head(self._extract_features(self.channel_proj(x)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = EarlyFusionSegmentationModel(num_classes=5, pretrained=False)
    model.eval()

    B, H = 2, 256
    x = torch.randn(B, 4, H, H)
    with torch.no_grad():
        out = model(x)

    expected = (B, 5, H, H)
    status = "PASS" if out.shape == expected else "FAIL"
    print(f"{status}  output shape: {tuple(out.shape)}  expected: {expected}")
