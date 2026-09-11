"""
Defect Detection Dashboard
Streamlit-based evaluation UI for Late Fusion MobileViT segmentation models.

Run with: streamlit run dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import io

import numpy as np
import streamlit as st
import torch
from PIL import Image

# Make sure the repo root is importable (for the common/ and models/ packages)
sys.path.insert(0, str(Path(__file__).parent))

import timm
from models.late_fusion_v1.model import LateFusionSegmentationModel, MultiModalDataPreprocessor
from models.late_fusion_v2.model import LateFusionSegmentationModelV2
from models.early_fusion_v2.model import EarlyFusionSegmentationModelV2
from models.CNN_VIT_SELF.model import CNNViTSelfSegmentationModel
from models.mobilenet_v4.model import MobileNetV4SegmentationModel
from common.heads import MobileViTSegmentationHead


class _LegacyEarlyFusionV2(torch.nn.Module):
    """
    Compatibility shim for early-fusion checkpoints saved before the ASPP
    decoder was introduced.  Matches the old seg_head=MobileViTSegmentationHead
    architecture so those checkpoints can still be loaded.
    """
    def __init__(self, num_classes: int = 5, hidden_dim: int = 256, in_channels: int = 4):
        super().__init__()
        self.channel_proj = torch.nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        self.backbone     = timm.create_model("mobilevitv2_100", pretrained=False)
        backbone_ch       = getattr(self.backbone, "num_features", 256)
        self.seg_head     = MobileViTSegmentationHead(backbone_ch, num_classes, hidden_dim)

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

# ─── Constants ────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent / "models"
IMAGE_SIZE = 256
NUM_CLASSES = 5

CLASS_NAMES = {0: "Background", 1: "Crack", 2: "Spall", 3: "Delamination", 4: "Moisture"}

# RGBA colors for each defect class (background kept transparent)
CLASS_COLORS_RGBA = {
    0: (0,   0,   0,   0),    # background – transparent
    1: (220, 50,  50,  200),  # crack       – red
    2: (255, 140, 0,   200),  # spall       – orange
    3: (240, 210, 20,  200),  # delamination– yellow
    4: (40,  100, 220, 200),  # moisture    – blue
}

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Defect Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Minimal custom CSS ───────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
    }
    .detected   { color: #a6e3a1; font-weight: 700; }
    .undetected { color: #6c7086; }
    .legend-dot {
        display: inline-block;
        width: 14px; height: 14px;
        border-radius: 50%;
        margin-right: 6px;
        vertical-align: middle;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def list_checkpoints() -> list[Path]:
    """Every *.pth under models/<name>/checkpoints/, across all model folders."""
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("*/checkpoints/*.pth"), key=lambda p: p.name)


def checkpoint_label(p: Path) -> str:
    """'<model_folder>/<filename>' — the model folder disambiguates identical filenames."""
    return f"{p.parent.parent.name}/{p.name}"


def is_cnn_vit_self(name: str) -> bool:
    return "cnn_vit_self" in name.lower()


def is_mobilenet_v4(name: str) -> bool:
    return "mobilenet_v4" in name.lower()


def is_early_fusion(name: str) -> bool:
    return "early_fusion" in name.lower()


def is_v2(name: str) -> bool:
    return "_v2_" in name.lower() and not is_early_fusion(name)


def detect_hidden_dim(state: dict) -> int:
    """Read hidden_dim from decoder weight shapes (handles all architectures)."""
    # Early fusion ASPP decoder
    if "decoder.project.0.weight" in state:
        return int(state["decoder.project.0.weight"].shape[0])
    # Early fusion legacy (MobileViTSegmentationHead)
    if "seg_head.decoder.0.0.weight" in state:
        return int(state["seg_head.decoder.0.0.weight"].shape[0])
    # Late fusion decoder
    if "decoder.rgb_dec1.0.weight" in state:
        return int(state["decoder.rgb_dec1.0.weight"].shape[0])
    return 256


@st.cache_resource(show_spinner="Loading model weights…")
def load_model(checkpoint_path: str):
    """Load and cache a model from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    hidden_dim = detect_hidden_dim(state)
    name = Path(checkpoint_path).name

    if is_cnn_vit_self(name):
        model = CNNViTSelfSegmentationModel(
            num_classes=NUM_CLASSES,
            hidden_dim=hidden_dim,
            in_channels=4,
        )
    elif is_mobilenet_v4(name):
        model = MobileNetV4SegmentationModel(
            num_classes=NUM_CLASSES,
            pretrained=False,
            hidden_dim=hidden_dim,
            in_channels=4,
        )
    elif is_early_fusion(name):
        if "seg_head.seg_head.weight" in state:
            # Checkpoint saved before ASPP decoder — use legacy architecture
            model = _LegacyEarlyFusionV2(
                num_classes=NUM_CLASSES,
                hidden_dim=hidden_dim,
                in_channels=4,
            )
        else:
            model = EarlyFusionSegmentationModelV2(
                num_classes=NUM_CLASSES,
                pretrained=False,
                hidden_dim=hidden_dim,
                in_channels=4,
            )
    elif is_v2(name):
        model = LateFusionSegmentationModelV2(
            num_classes=NUM_CLASSES,
            backbone_name="mobilevitv2_100",
            pretrained=False,
            hidden_dim=hidden_dim,
        )
    else:
        model = LateFusionSegmentationModel(
            num_classes=NUM_CLASSES,
            backbone_name="mobilevit_s",
            pretrained=False,
            hidden_dim=hidden_dim,
        )

    # Strip loss-function weights — not needed for inference
    state = {k: v for k, v in state.items() if not k.startswith("criterion")}
    model.load_state_dict(state, strict=False)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, device, ckpt


def pil_to_rgb_tensor(pil_img: Image.Image) -> torch.Tensor:
    """PIL → normalised [1, 3, H, W] tensor (ImageNet stats)."""
    img = pil_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return MultiModalDataPreprocessor.preprocess_rgb(t)


def pil_to_irt_tensor(pil_img: Image.Image) -> torch.Tensor:
    """PIL → min-max normalised [1, 1, H, W] tensor."""
    img = pil_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)   # [1, 3, H, W]
    return MultiModalDataPreprocessor.preprocess_irt(t)        # → [1, 1, H, W]


def zero_rgb_tensor() -> torch.Tensor:
    """ImageNet-normalised zero tensor standing in for a missing RGB frame."""
    t = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    return MultiModalDataPreprocessor.preprocess_rgb(t)


def zero_irt_tensor() -> torch.Tensor:
    return torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    device: str,
    rgb_t: torch.Tensor,
    irt_t: torch.Tensor,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    mask  : [H, W] int, predicted class (defects below threshold → 0)
    probs : [C, H, W] float, per-class probability maps
    """
    if hasattr(model, "channel_proj") or isinstance(model, CNNViTSelfSegmentationModel):
        # Early fusion (or the from-scratch CNN-ViT-Self model, which takes
        # the 4-channel input directly with no separate projection layer):
        # concatenate RGB [1,3,H,W] + IRT [1,1,H,W] → [1,4,H,W]
        fused = torch.cat([rgb_t, irt_t], dim=1).to(device)
        logits = model(fused)
    else:
        logits = model(rgb_t.to(device), irt_t.to(device))   # [1, C, H, W]
    probs = torch.softmax(logits, dim=1).squeeze(0)       # [C, H, W]
    mask  = probs.argmax(dim=0)                           # [H, W]

    # Suppress predictions below the confidence threshold
    max_prob = probs.max(dim=0).values                    # [H, W]
    mask[max_prob < threshold] = 0

    return mask.cpu().numpy().astype(np.int32), probs.cpu().numpy()


def build_overlay(base_rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Blend a color-coded segmentation mask onto a uint8 RGB image."""
    out = base_rgb.copy().astype(np.float32)
    for cls_id, rgba in CLASS_COLORS_RGBA.items():
        if cls_id == 0:
            continue
        where = mask == cls_id
        if not where.any():
            continue
        color = np.array(rgba[:3], dtype=np.float32)
        out[where] = (1 - alpha) * out[where] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def class_stats(mask: np.ndarray, probs: np.ndarray) -> list[dict]:
    total = mask.size
    rows = []
    for cls_id in range(1, NUM_CLASSES):
        pixels   = int((mask == cls_id).sum())
        coverage = pixels / total * 100
        mean_p   = float(probs[cls_id].mean())
        rows.append(
            dict(
                cls_id=cls_id,
                name=CLASS_NAMES[cls_id],
                pixels=pixels,
                coverage=coverage,
                mean_conf=mean_p,
            )
        )
    return rows


def legend_html() -> str:
    items = []
    for cls_id, name in CLASS_NAMES.items():
        if cls_id == 0:
            continue
        r, g, b, _ = CLASS_COLORS_RGBA[cls_id]
        items.append(
            f'<span class="legend-dot" style="background:rgb({r},{g},{b})"></span>'
            f'<span style="margin-right:16px">{name}</span>'
        )
    return "".join(items)


def resize_for_display(pil_img: Image.Image, max_dim: int = 512) -> Image.Image:
    w, h = pil_img.size
    scale = min(max_dim / w, max_dim / h, 1.0)
    return pil_img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Defect Detection")
    st.caption("MobileViT Late Fusion · UAV Inspection")
    st.markdown("---")

    # Checkpoint selection
    checkpoints = list_checkpoints()
    if not checkpoints:
        st.error(f"No .pth files found under `{MODELS_DIR}/*/checkpoints/`")
        st.stop()

    ckpt_labels = [checkpoint_label(p) for p in checkpoints]
    selected_label = st.selectbox("Checkpoint", ckpt_labels, index=len(ckpt_labels) - 1)
    selected_path = str(checkpoints[ckpt_labels.index(selected_label)])

    # Show quick metadata without loading full model
    try:
        meta = torch.load(selected_path, map_location="cpu", weights_only=False)
        col1, col2 = st.columns(2)
        col1.metric("Epoch", meta.get("epoch", "—"))
        best = meta.get("best_metric")
        col2.metric("Best mIoU", f"{best:.4f}" if isinstance(best, float) else "—")
        metrics = meta.get("metrics", {})
        if metrics:
            extra = {k: f"{v:.4f}" if isinstance(v, float) else v
                     for k, v in metrics.items() if k != "epoch"}
            with st.expander("Checkpoint metrics"):
                for k, v in extra.items():
                    st.write(f"**{k}**: {v}")
        del meta
    except Exception as exc:
        st.warning(f"Could not read metadata: {exc}")

    st.markdown("---")

    # Modality selection
    modality = st.radio(
        "Input Modality",
        ["RGB + IR (Fusion)", "RGB only", "IR only"],
        index=0,
    )

    st.markdown("---")

    # Inference settings
    threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.50, 0.05,
                          help="Pixels with max class probability below this are set to background.")
    overlay_alpha = st.slider("Overlay Opacity", 0.1, 1.0, 0.55, 0.05)

# ─── Main panel ───────────────────────────────────────────────────────────────

st.header("Image Input")

need_rgb = modality in ("RGB + IR (Fusion)", "RGB only")
need_irt = modality in ("RGB + IR (Fusion)", "IR only")

upload_cols = st.columns(2 if (need_rgb and need_irt) else 1)
rgb_file = irt_file = None

if need_rgb:
    col = upload_cols[0]
    rgb_file = col.file_uploader("RGB Image", type=["png", "jpg", "jpeg", "tif", "tiff"],
                                  key="rgb_up")

if need_irt:
    col = upload_cols[-1]
    irt_file = col.file_uploader("IR (Thermal) Image", type=["png", "jpg", "jpeg", "tif", "tiff"],
                                  key="irt_up")

# Preview uploaded images
preview_cols = []
rgb_pil = irt_pil = None

if rgb_file:
    rgb_pil = Image.open(io.BytesIO(rgb_file.read())).convert("RGB")
    rgb_file.seek(0)
if irt_file:
    irt_pil = Image.open(io.BytesIO(irt_file.read())).convert("RGB")
    irt_file.seek(0)

ready = (
    (need_rgb and need_irt and rgb_pil and irt_pil)
    or (need_rgb and not need_irt and rgb_pil)
    or (need_irt and not need_rgb and irt_pil)
)

if rgb_pil or irt_pil:
    prev_cols = st.columns(2 if (rgb_pil and irt_pil) else 1)
    if rgb_pil:
        prev_cols[0].image(resize_for_display(rgb_pil), caption="RGB Input", use_container_width=True)
    if irt_pil:
        prev_cols[-1].image(resize_for_display(irt_pil), caption="IR Input", use_container_width=True)

st.markdown("---")

# ─── Run inference ────────────────────────────────────────────────────────────

run_btn = st.button("Run Inference", type="primary", disabled=not ready)

if run_btn and ready:
    with st.spinner("Loading model…"):
        model, device, _ = load_model(selected_path)

    with st.spinner("Running inference…"):
        # Build input tensors
        if rgb_pil:
            rgb_t = pil_to_rgb_tensor(rgb_pil)
        else:
            rgb_t = zero_rgb_tensor()

        if irt_pil:
            irt_t = pil_to_irt_tensor(irt_pil)
        else:
            irt_t = zero_irt_tensor()

        mask, probs = run_inference(model, device, rgb_t, irt_t, threshold)

    # ── Result visualisation ─────────────────────────────────────────────────

    st.header("Results")

    # Legend
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.markdown("")

    # Choose display base: prefer RGB, fall back to IRT
    base_pil = rgb_pil if rgb_pil else irt_pil
    base_arr = np.array(base_pil.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR))

    overlay_arr = build_overlay(base_arr, mask, overlay_alpha)
    overlay_pil = Image.fromarray(overlay_arr)

    # Pure mask image (colored, no alpha blending)
    mask_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for cls_id, rgba in CLASS_COLORS_RGBA.items():
        mask_img[mask == cls_id] = rgba[:3]
    mask_pil = Image.fromarray(mask_img)

    res_cols = st.columns(3)
    res_cols[0].image(resize_for_display(base_pil),    caption="Original",           use_container_width=True)
    res_cols[1].image(resize_for_display(overlay_pil), caption="Overlay",            use_container_width=True)
    res_cols[2].image(resize_for_display(mask_pil),    caption="Segmentation Mask",  use_container_width=True)

    # ── Per-class statistics ──────────────────────────────────────────────────

    st.subheader("Detection Summary")
    stats = class_stats(mask, probs)
    any_detected = any(s["pixels"] > 0 for s in stats)

    if not any_detected:
        st.info("No defects detected above the confidence threshold.")
    else:
        tbl_cols = st.columns([2, 1, 2, 2])
        tbl_cols[0].markdown("**Class**")
        tbl_cols[1].markdown("**Detected**")
        tbl_cols[2].markdown("**Coverage**")
        tbl_cols[3].markdown("**Mean Confidence**")

        for s in stats:
            detected = s["pixels"] > 0
            css = "detected" if detected else "undetected"
            icon = "✓" if detected else "✗"
            c0, c1, c2, c3 = st.columns([2, 1, 2, 2])
            c0.markdown(f'<span class="{css}">{s["name"]}</span>', unsafe_allow_html=True)
            c1.markdown(f'<span class="{css}">{icon}</span>',      unsafe_allow_html=True)
            c2.markdown(
                f'<span class="{css}">{s["coverage"]:.2f}%</span>' if detected
                else '<span class="undetected">—</span>',
                unsafe_allow_html=True,
            )
            c3.markdown(f'<span class="{css}">{s["mean_conf"]:.3f}</span>', unsafe_allow_html=True)

    # ── Download ──────────────────────────────────────────────────────────────

    st.markdown("---")
    dl_col1, dl_col2 = st.columns(2)

    buf_ov = io.BytesIO()
    overlay_pil.save(buf_ov, format="PNG")
    dl_col1.download_button("Download Overlay", buf_ov.getvalue(),
                             file_name="overlay.png", mime="image/png")

    buf_mk = io.BytesIO()
    mask_pil.save(buf_mk, format="PNG")
    dl_col2.download_button("Download Mask", buf_mk.getvalue(),
                             file_name="mask.png", mime="image/png")

elif run_btn and not ready:
    st.warning("Please upload the required image(s) before running inference.")
