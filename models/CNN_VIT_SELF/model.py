"""
CNN-ViT-Self — a from-scratch CNN + self-attention hybrid for early-fusion
defect segmentation.

Every other model in this repo builds on a `timm` pretrained MobileViT(v2)
backbone. This one does not: the whole backbone (convolutional stem, CNN
stages, and self-attention blocks) is defined and trained from scratch here,
with no ImageNet-pretrained weights. Two consequences of that:

  1. It can consume the 4-channel fused RGBT input directly in its first
     convolution — no learnable 1x1 "channel projection" down to 3 channels
     is needed (that projection exists in models/early_fusion_v1 and
     models/early_fusion_v2 specifically to keep those inputs compatible
     with a pretrained 3-channel backbone).
  2. It will need more training data/epochs to reach the same accuracy as
     the pretrained models, since it has no prior visual knowledge at
     initialization.

Architecture
------------
  Fused RGBT [B, 4, H, W]
      -> Conv stem                         (4   -> 32,  H    -> H/2)
      -> CNNStage  (local conv only)       (32  -> 64,  H/2  -> H/4)
      -> CNNStage  (local conv only)       (64  -> 128, H/4  -> H/8)
      -> CNNViTBlock (local conv + attn)   (128 -> 256, H/8  -> H/16)
      -> CNNViTBlock (local conv + attn)   (256 -> 384, H/16 -> H/32)
      -> common.heads.MobileViTSegmentationHead (5x 2x upsample -> H)
      -> logits [B, num_classes, H, W]

Early stages stay pure-CNN (no attention) because self-attention over a
high-resolution token grid is expensive; attention is only applied once the
grid is small enough to be cheap (H/16 and H/32), which is exactly the
"local features first, global reasoning once downsampled" idea MobileViT
itself is built on — reimplemented here with plain nn.MultiheadAttention
instead of a pretrained ViT.

Class IDs (shared across all models):
  0  background
  1  crack
  2  spall
  3  delamination  (reserved; no annotations yet)
  4  moisture
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.heads import MobileViTSegmentationHead
from common.losses import FocalDiceLoss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2D sinusoidal positional encoding (computed fresh per forward pass, so the
# model isn't locked to one fixed image_size the way a learned embedding
# table would be)
# ---------------------------------------------------------------------------

def _sinusoidal_positional_encoding(h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
    """Returns [h*w, dim] positional encodings for an h x w token grid."""
    if dim % 4 != 0:
        raise ValueError(f"positional encoding dim must be divisible by 4, got {dim}")
    d_quarter = dim // 4
    div_term = torch.exp(
        torch.arange(0, d_quarter, device=device, dtype=dtype) * (-math.log(10000.0) / d_quarter)
    )

    y_pos = torch.arange(h, device=device, dtype=dtype).unsqueeze(1)  # [h, 1]
    x_pos = torch.arange(w, device=device, dtype=dtype).unsqueeze(1)  # [w, 1]

    pe_y = torch.zeros(h, d_quarter * 2, device=device, dtype=dtype)
    pe_y[:, 0::2] = torch.sin(y_pos * div_term)
    pe_y[:, 1::2] = torch.cos(y_pos * div_term)

    pe_x = torch.zeros(w, d_quarter * 2, device=device, dtype=dtype)
    pe_x[:, 0::2] = torch.sin(x_pos * div_term)
    pe_x[:, 1::2] = torch.cos(x_pos * div_term)

    pe_y = pe_y.unsqueeze(1).expand(h, w, d_quarter * 2)   # [h, w, dim/2]
    pe_x = pe_x.unsqueeze(0).expand(h, w, d_quarter * 2)   # [h, w, dim/2]
    pe = torch.cat([pe_y, pe_x], dim=-1)                    # [h, w, dim]
    return pe.reshape(h * w, dim)


# ---------------------------------------------------------------------------
# Pure-CNN stage (depthwise-separable "inverted residual", MobileNet-style)
# ---------------------------------------------------------------------------

class CNNStage(nn.Module):
    """Local-only feature extraction + 2x downsample. No self-attention."""

    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True):
        super().__init__()
        stride = 2 if downsample else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.ReLU6(inplace=True)
        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x) if self.skip is not None else x
        return self.act(self.block(x) + identity)


# ---------------------------------------------------------------------------
# CNN + self-attention hybrid block — the "ViT" half of the hybrid
# ---------------------------------------------------------------------------

class CNNViTBlock(nn.Module):
    """
    Local branch (depthwise-separable conv, handles the 2x downsample) whose
    output is also tokenized and passed through one pre-norm Transformer
    encoder layer (multi-head self-attention + MLP) for global context, then
    fused back into the spatial map.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        downsample: bool = True,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if out_ch % num_heads != 0:
            raise ValueError(f"out_ch ({out_ch}) must be divisible by num_heads ({num_heads})")

        stride = 2 if downsample else 1
        self.local = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.local_act = nn.ReLU6(inplace=True)
        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.norm1 = nn.LayerNorm(out_ch)
        self.attn = nn.MultiheadAttention(
            out_ch, num_heads, dropout=attn_dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(out_ch)
        hidden = int(out_ch * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(out_ch, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_ch),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x) if self.skip is not None else x
        local_out = self.local_act(self.local(x) + identity)   # [B, out_ch, H', W']

        B, C, H, W = local_out.shape
        tokens = local_out.flatten(2).transpose(1, 2)          # [B, N, C]
        pos = _sinusoidal_positional_encoding(H, W, C, tokens.device, tokens.dtype)
        tokens = tokens + pos.unsqueeze(0)

        attn_in = self.norm1(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))

        global_out = tokens.transpose(1, 2).reshape(B, C, H, W)
        return self.fuse(global_out) + local_out


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class CNNViTSelfSegmentationModel(nn.Module):
    """
    From-scratch CNN-ViT hybrid for 4-channel early-fusion segmentation.

    Input tensor shape : [B, 4, H, W]   (fused RGB+IR, H and W must be
                          divisible by 32 — the same requirement every other
                          model here has via its stride-32 backbone)
    Output tensor shape: [B, num_classes, H, W]
    """

    def __init__(
        self,
        num_classes: int = 5,
        in_channels: int = 4,
        hidden_dim: int = 256,
        stem_channels: int = 32,
        stage_channels: tuple[int, int, int, int] = (64, 128, 256, 384),
        num_heads: int = 4,
        class_weights: list | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        c1, c2, c3, c4 = stage_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU6(inplace=True),
        )  # H -> H/2

        self.stage1 = CNNStage(stem_channels, c1, downsample=True)               # H/2  -> H/4
        self.stage2 = CNNStage(c1, c2, downsample=True)                          # H/4  -> H/8
        self.stage3 = CNNViTBlock(c2, c3, num_heads=num_heads, downsample=True)  # H/8  -> H/16
        self.stage4 = CNNViTBlock(c3, c4, num_heads=num_heads, downsample=True)  # H/16 -> H/32

        self.seg_head = MobileViTSegmentationHead(c4, num_classes, hidden_dim)

        if class_weights is None:
            # Same starting point as early_fusion_v2's data-driven default
            # fallback — override with common.dataset_utils.compute_class_weights
            # once real annotation statistics are available.
            class_weights = [0.5, 3.0, 2.0, 1.0, 10.0]
        weights = torch.tensor(class_weights[:num_classes], dtype=torch.float32)
        self.criterion = FocalDiceLoss(weights, num_classes)

        n_params = sum(p.numel() for p in self.parameters())
        logger.info("CNNViTSelfSegmentationModel initialised | %.2fM parameters", n_params / 1e6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.seg_head(x)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model = CNNViTSelfSegmentationModel(num_classes=5)
    model.eval()

    B, H = 2, 256
    x = torch.randn(B, 4, H, H)
    with torch.no_grad():
        out = model(x)

    expected = (B, 5, H, H)
    status = "PASS" if out.shape == expected else "FAIL"
    print(f"{status}  output shape: {tuple(out.shape)}  expected: {expected}")
