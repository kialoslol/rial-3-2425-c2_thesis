"""
Late Fusion MobileViT Model for Multi-Modal Defect Segmentation
UAV-Based Infrared Thermography + RGB with CNN-ViT Hybrid Backbone

Architecture overview
─────────────────────
RGB  image [B, 3, H, W] ──► MobileViT-S backbone ──► [B, C, h, w]  ┐
                                                                       ├─► LateFusionDecoder ──► [B, num_classes, H, W]
IRT  image [B, 1, H, W] ──► MobileViT-S backbone ──► [B, C, h, w]  ┘
         (1-ch adapted)

Decoder: 5 × 2× bilinear upsample stages → full input resolution.
Fusion:  ModalityAttention at decoder levels 1 and 2 (lowest resolution,
         where cross-modal context matters most); single fused pathway
         for levels 3-5.

Class IDs (consistent across all modules):
  0  background
  1  crack
  2  spall
  3  delamination  (reserved; annotations will be added in future datasets)
  4  moisture
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modality Attention
# ---------------------------------------------------------------------------

class ModalityAttention(nn.Module):
    """
    Learn per-spatial-location importance weights for two feature maps.

    Given RGB and IRT feature maps of the same shape [B, C, H, W], produce
    soft weights that emphasise which modality is more informative at each
    pixel, then return the weighted features.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 2, kernel_size=1),
            nn.Softmax(dim=1),           # [B, 2, H, W]
        )

    def forward(
        self, rgb_feat: torch.Tensor, irt_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([rgb_feat, irt_feat], dim=1)
        weights  = self.attention(combined)               # [B, 2, H, W]
        return rgb_feat * weights[:, 0:1], irt_feat * weights[:, 1:2]


# ---------------------------------------------------------------------------
# Decoder block helper
# ---------------------------------------------------------------------------

def _dec_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv → BN → ReLU → 2× bilinear upsample."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
    )


def _fuse_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """1×1 Conv → BN → ReLU used after concatenating two modality branches."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# Late Fusion Decoder
# ---------------------------------------------------------------------------

class LateFusionDecoder(nn.Module):
    """
    Progressive upsampling decoder with two levels of cross-modal attention.

    Backbone output: [B, backbone_ch, h, w]  (h = H/32 for mobilevit_s)

    Upsampling chain (5 × 2× = 32× for stride-32 backbone):
      L1  backbone_ch → hidden       h   → 2h    dual-branch + attention
      L2  hidden      → hidden//2    2h  → 4h    dual-branch + attention
      L3  hidden//2   → hidden//4    4h  → 8h    single fused
      L4  hidden//4   → hidden//4    8h  → 16h   single fused
      L5  hidden//4   → hidden//4    16h → 32h   single fused  (= H)
      seg hidden//4   → num_classes  (1×1, no upsample)

    With hidden_dim=256, backbone_ch=640, H=512:
      h=16, final output 512×512 matching the input.
    """

    def __init__(self, backbone_ch: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        d  = hidden_dim
        d2 = hidden_dim // 2
        d4 = hidden_dim // 4

        # ── Level 1: dual-pathway (RGB / IRT), 2× upsample ─────────────────
        self.rgb_dec1  = _dec_block(backbone_ch, d)
        self.irt_dec1  = _dec_block(backbone_ch, d)
        self.attn1     = ModalityAttention(d)
        self.fuse1     = _fuse_block(d * 2, d)

        # ── Level 2: dual-pathway from shared fused, 2× upsample ───────────
        self.rgb_dec2  = _dec_block(d, d2)
        self.irt_dec2  = _dec_block(d, d2)
        self.attn2     = ModalityAttention(d2)
        self.fuse2     = _fuse_block(d2 * 2, d2)

        # ── Levels 3-5: single fused pathway, 2× upsample each ─────────────
        self.dec3 = _dec_block(d2, d4)
        self.dec4 = _dec_block(d4, d4)
        self.dec5 = _dec_block(d4, d4)

        # ── Segmentation head (no extra upsample) ──────────────────────────
        self.seg_head = nn.Conv2d(d4, num_classes, kernel_size=1)

    def forward(
        self, rgb_feat: torch.Tensor, irt_feat: torch.Tensor
    ) -> torch.Tensor:
        # L1 — separate pathways + attention fusion
        rgb_x = self.rgb_dec1(rgb_feat)
        irt_x = self.irt_dec1(irt_feat)
        rgb_w, irt_w = self.attn1(rgb_x, irt_x)
        x = self.fuse1(torch.cat([rgb_w, irt_w], dim=1))

        # L2 — apply two learned projections on the same fused tensor,
        #       then re-fuse with attention (adds model capacity)
        rgb_y = self.rgb_dec2(x)
        irt_y = self.irt_dec2(x)
        rgb_w2, irt_w2 = self.attn2(rgb_y, irt_y)
        x = self.fuse2(torch.cat([rgb_w2, irt_w2], dim=1))

        # L3-L5 — single fused pathway
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)

        return self.seg_head(x)


# ---------------------------------------------------------------------------
# Full Late Fusion Segmentation Model
# ---------------------------------------------------------------------------

class LateFusionSegmentationModel(nn.Module):
    """
    Dual-backbone MobileViT-S late fusion segmentation model.

    Inputs
    ------
    rgb_image     : [B, 3, H, W]  — RGB (ImageNet-normalised)
    thermal_image : [B, 1, H, W]  — IRT (min-max normalised to [0,1])

    Output
    ------
    logits : [B, num_classes, H, W]

    The thermal backbone's first Conv2d is adapted from 3-channel to 1-channel
    input by summing the pretrained RGB filters (channel-sum initialisation
    preserves the effective receptive field energy).
    """

    def __init__(
        self,
        num_classes: int = 5,
        backbone_name: str = "mobilevit_s",
        pretrained: bool = True,
        hidden_dim: int = 256,
        class_weights: list | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        logger.info("Initialising LateFusionSegmentationModel")
        logger.info("  backbone=%s  num_classes=%d  pretrained=%s",
                    backbone_name, num_classes, pretrained)

        # ── RGB backbone ────────────────────────────────────────────────────
        self.rgb_backbone = self._load_backbone(backbone_name, pretrained)

        # ── IRT backbone (separate weights, adapted to 1-channel input) ─────
        self.irt_backbone = self._load_backbone(backbone_name, pretrained)
        self._adapt_to_single_channel(self.irt_backbone)

        # ── Determine backbone output channels ──────────────────────────────
        backbone_ch = getattr(self.rgb_backbone, "num_features", 640)
        logger.info("  backbone output channels: %d", backbone_ch)

        # ── Decoder ─────────────────────────────────────────────────────────
        self.decoder = LateFusionDecoder(backbone_ch, num_classes, hidden_dim)
        logger.info("  decoder hidden_dim=%d  total upsample=32×", hidden_dim)

        # ── Loss ─────────────────────────────────────────────────────────────
        # Default class weights: downweight background, upweight rare defects.
        # bg=0.2, crack=3.0, spall=2.0, delamination=1.0 (no data), moisture=3.0
        if class_weights is None:
            class_weights = [0.2, 3.0, 2.0, 1.0, 3.0]
        weights = torch.tensor(class_weights[:num_classes], dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)

    # ------------------------------------------------------------------

    @staticmethod
    def _load_backbone(name: str, pretrained: bool) -> nn.Module:
        try:
            m = timm.create_model(name, pretrained=pretrained, features_only=False)
            logger.info("  ✓ Backbone loaded: %s", name)
            return m
        except Exception:
            logger.warning("  ⚠ %s unavailable, falling back to mobilenetv3_small_100", name)
            return timm.create_model("mobilenetv3_small_100", pretrained=pretrained)

    @staticmethod
    def _adapt_to_single_channel(backbone: nn.Module) -> None:
        """
        Replace the first Conv2d in `backbone` with a 1-channel equivalent.

        The new weights are the *sum* of the original three channel-filters,
        which preserves the magnitude of the dot-product for a normalised
        single-channel input.
        """
        first_name: str | None = None
        first_conv: nn.Conv2d | None = None

        for name, module in backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                first_name = name
                first_conv = module
                break

        if first_conv is None or first_name is None:
            logger.warning("  ⚠ No Conv2d found in backbone; thermal adaptation skipped")
            return

        # Sum RGB filters → single filter
        new_weight = first_conv.weight.data.sum(dim=1, keepdim=True)   # [out, 1, kH, kW]

        new_conv = nn.Conv2d(
            1,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            dilation=first_conv.dilation,
            groups=first_conv.groups,
            bias=first_conv.bias is not None,
        )
        new_conv.weight.data.copy_(new_weight)
        if first_conv.bias is not None:
            new_conv.bias.data.copy_(first_conv.bias.data)

        # Navigate to the parent module and replace the attribute
        parts  = first_name.split(".")
        parent = backbone
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_conv)

        logger.info("  ✓ IRT backbone first Conv2d adapted: %s → 1-channel input", first_name)

    # ------------------------------------------------------------------

    def _extract_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward_features and ensure the result is [B, C, H, W].

        Some timm models return [B, N, C] (sequence format from ViT stages).
        We reshape back to a spatial map assuming a square token grid.
        """
        feats = backbone.forward_features(x)

        if feats.dim() == 4:
            return feats                             # already [B, C, H, W]

        if feats.dim() == 3:
            # [B, N, C] → [B, C, H, W]  (N must be a perfect square)
            B, N, C = feats.shape
            h = w = int(N ** 0.5)
            if h * w != N:
                raise ValueError(
                    f"Cannot reshape feature sequence of length {N} "
                    f"into a square spatial map."
                )
            return feats.permute(0, 2, 1).reshape(B, C, h, w)

        raise ValueError(f"Unexpected feature tensor dim: {feats.dim()}")

    # ------------------------------------------------------------------

    def forward(
        self, rgb_image: torch.Tensor, thermal_image: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            rgb_image     : [B, 3, H, W]
            thermal_image : [B, 1, H, W]

        Returns:
            logits : [B, num_classes, H, W]
        """
        rgb_feats = self._extract_features(self.rgb_backbone, rgb_image)
        irt_feats = self._extract_features(self.irt_backbone, thermal_image)
        return self.decoder(rgb_feats, irt_feats)


# ---------------------------------------------------------------------------
# Preprocessing utilities (kept for inference convenience)
# ---------------------------------------------------------------------------

class MultiModalDataPreprocessor:
    """Static helpers for normalising individual modality images at inference."""

    @staticmethod
    def preprocess_rgb(rgb: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(rgb)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(rgb)
        return (rgb - mean) / (std + 1e-7)

    @staticmethod
    def preprocess_irt(irt: torch.Tensor) -> torch.Tensor:
        if irt.shape[1] == 3:
            irt = irt.mean(dim=1, keepdim=True)
        lo = irt.flatten(1).min(dim=1).values.view(-1, 1, 1, 1)
        hi = irt.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
        return (irt - lo) / (hi - lo + 1e-7)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = LateFusionSegmentationModel(
        num_classes=5,
        backbone_name="mobilevit_s",
        pretrained=False,
    )
    model.eval()

    B, H = 2, 512
    rgb     = torch.randn(B, 3, H, H)
    thermal = torch.randn(B, 1, H, H)

    with torch.no_grad():
        out = model(rgb, thermal)

    expected = (B, 5, H, H)
    status   = "✓ PASS" if out.shape == expected else "✗ FAIL"
    print(f"{status}  output shape: {tuple(out.shape)}  expected: {expected}")
