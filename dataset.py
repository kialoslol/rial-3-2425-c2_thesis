"""
Dataset loaders for defect segmentation.

COCOSegmentationDataset — Early Fusion (4-channel fused TIFF: RGB+IR) or plain RGB segmentation.
MultiModalDataset       — Dual-modal (separate RGB PNG + IRT JPG) for Late Fusion model.

Both classes use albumentations so that *all* geometric augmentations are applied
identically to the image and the segmentation mask, preventing label misalignment.

Class IDs (shared across all modules):
  0  background
  1  crack
  2  spall
  3  delamination  (reserved; no annotations yet)
  4  moisture
"""

from __future__ import annotations

import os

import logging
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset


os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
logger = logging.getLogger(__name__)

# ImageNet statistics for RGB normalisation
_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_augmentation_pipeline(image_size: int, augment: bool) -> A.Compose:
    """
    Albumentations pipeline applied synchronously to image(s) + mask.

    Geometric transforms are only added when augment=True; photometric
    transforms are only applied to colour images (not IRT).
    """
    spatial = [
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.85, 1.15),
            rotate=(-20, 20),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            fill_mask=0,
            p=0.5,
        ),
        # ElasticTransform: deforms crack-like thin structures realistically
        A.ElasticTransform(alpha=80, sigma=8, p=0.2),
        # GridDistortion: simulates surface warp from different camera angles
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
    ] if augment else [
        A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
    ]

    photometric = [
        # Stronger brightness/contrast — moisture is highly lighting-dependent
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.6),
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1, p=0.5),
        # CLAHE sharpens local contrast; helps distinguish moisture patches
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.4),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        # Sharpen to recover edge detail lost after blur/noise
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.3),
    ] if augment else []

    return A.Compose(spatial + photometric)


def _normalize_rgb(img: np.ndarray) -> np.ndarray:
    """uint8 HWC → float32 HWC, ImageNet-normalised."""
    return (img.astype(np.float32) / 255.0 - _RGB_MEAN) / _RGB_STD


def _normalize_irt(img: np.ndarray) -> np.ndarray:
    """Uint8/float grayscale HW → float32 HW in [0, 1] via min-max."""
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-7)


def _normalize_fused(rgb: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Normalize RGB [H,W,3] and IR [H,W] separately and concatenate into [H,W,4]."""
    rgb_norm = _normalize_rgb(rgb)                    # [H, W, 3] float32
    ir_norm  = _normalize_irt(ir)[..., np.newaxis]   # [H, W, 1] float32
    return np.concatenate([rgb_norm, ir_norm], axis=2)  # [H, W, 4]


def _build_mask(coco: COCO, img_id: int, h: int, w: int) -> np.ndarray:
    """Render COCO polygon segmentation annotations into a (H, W) uint8 mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
        cat_id = int(ann["category_id"])
        for seg in ann.get("segmentation", []):
            pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], cat_id)
    return mask


def _find_image(directory: Path, stem: str) -> Path:
    """Return the first existing file matching <stem>.<ext> in directory."""
    for ext in (".png", ".tiff", ".tif", ".jpg", ".jpeg"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No image found for stem '{stem}' in {directory}. "
        f"Checked: .png .tiff .tif .jpg .jpeg"
    )


# ---------------------------------------------------------------------------
# Single-modal dataset  (fused images)
# ---------------------------------------------------------------------------

class COCOSegmentationDataset(Dataset):
    """
    Loads fused TIFF images (RGB+IR) referenced by a COCO JSON file.

    Always returns a [4, H, W] float32 tensor so every batch is uniform:
      channels 0-2 : RGB, ImageNet-normalised
      channel  3   : IR,  min-max normalised to [0, 1]
    If the source image has fewer than 4 channels the IR channel is filled
    with zeros so the batch shape stays consistent.

    Mask/image dimension mismatches (JSON dims ≠ actual TIFF dims) are
    corrected automatically. Bad files are skipped with a warning and the
    next valid item is returned instead.

    annotation_file: direct path to the COCO JSON (absolute, or relative to CWD).
                     It is NOT joined with root_dir.
    root_dir:        base directory used to resolve image file_name paths from the JSON.
    """

    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        image_size: int = 512,
        augment: bool = True,
    ):
        self.root_dir   = Path(root_dir)
        self.image_size = image_size

        self.coco      = COCO(str(annotation_file))
        self.image_ids = list(self.coco.imgs.keys())

        # Single fused pipeline: always treats RGB + IR as two image targets
        base = _build_augmentation_pipeline(image_size, augment)
        self.transform_fused = A.Compose(
            base.transforms,
            additional_targets={"irt": "image"},
        )

        # Pre-scan: warn about any images that cannot be found on disk
        missing = [
            self.coco.imgs[i]["file_name"]
            for i in self.image_ids
            if not (self.root_dir / self.coco.imgs[i]["file_name"]).exists()
        ]
        if missing:
            logger.warning(
                "COCOSegmentationDataset: %d image(s) not found on disk:\n  %s",
                len(missing), "\n  ".join(missing[:10]),
            )

        logger.info(
            "COCOSegmentationDataset | %d images | augment=%s | %s",
            len(self.image_ids), augment, annotation_file,
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        # Try up to len(dataset) neighbours before giving up
        for attempt in range(len(self)):
            try:
                return self._load_item((idx + attempt) % len(self))
            except Exception as exc:
                logger.warning(
                    "Skipping item %d (%s): %s",
                    (idx + attempt) % len(self),
                    self.coco.imgs[self.image_ids[(idx + attempt) % len(self)]]["file_name"],
                    exc,
                )
        raise RuntimeError("No valid items could be loaded from the dataset.")

    def _load_item(self, idx: int) -> dict:
        img_id   = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = self.root_dir / img_info["file_name"]

        if not img_path.exists():
            raise FileNotFoundError(f"File not found: {img_path}")

        raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise OSError(f"OpenCV could not decode: {img_path}")

        # ── Normalise array rank ────────────────────────────────────────────
        if raw.ndim == 2:
            raw = raw[:, :, np.newaxis]   # [H, W] → [H, W, 1]

        actual_h, actual_w = raw.shape[:2]

        # ── Build mask; fix dimension mismatch between JSON and actual file ─
        json_h = int(img_info.get("height") or actual_h)
        json_w = int(img_info.get("width")  or actual_w)
        mask = _build_mask(self.coco, img_id, json_h, json_w)
        if (json_h, json_w) != (actual_h, actual_w):
            mask = cv2.resize(
                mask, (actual_w, actual_h), interpolation=cv2.INTER_NEAREST
            )

        # ── Extract BGR and IR; always produce 4-channel output ─────────────
        if raw.shape[2] >= 4:
            bgr = raw[:, :, :3]
            ir  = raw[:, :, 3]
        elif raw.shape[2] == 3:
            bgr = raw
            ir  = np.zeros((actual_h, actual_w), dtype=np.uint8)  # no IR → zeros
        else:  # 1-channel grayscale
            bgr = cv2.cvtColor(raw[:, :, 0], cv2.COLOR_GRAY2BGR)
            ir  = np.zeros((actual_h, actual_w), dtype=np.uint8)

        rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        irt_3ch = np.stack([ir, ir, ir], axis=-1)   # albumentations needs HWC

        # ── Synchronised augmentation ────────────────────────────────────────
        out     = self.transform_fused(image=rgb, irt=irt_3ch, mask=mask)
        rgb_aug = out["image"]
        ir_aug  = out["irt"][:, :, 0]
        mask    = out["mask"]

        # ── Normalise & convert to tensors ───────────────────────────────────
        img_t  = torch.from_numpy(
            _normalize_fused(rgb_aug, ir_aug).transpose(2, 0, 1)
        ).float()                                        # [4, H, W]
        mask_t = torch.from_numpy(mask.astype(np.int64))  # [H, W]

        return {"image": img_t, "mask": mask_t, "image_id": img_id}


# ---------------------------------------------------------------------------
# Dual-modal dataset  (RGB + IRT)  for the Late Fusion model
# ---------------------------------------------------------------------------

class MultiModalDataset(Dataset):
    """
    Loads *separate* RGB and IRT images for the Late Fusion model.

    The COCO JSON stores image names like 'fuse_Binondo/000000.tiff'.
    The stem ('000000') is used to locate the corresponding files in
    `rgb_dir` (RGB_FITTED/) and `irt_dir` (IRT_FITTED/).

    All augmentation — including geometric transforms — is applied
    synchronously to the RGB image, IRT image, and segmentation mask via
    albumentations' `additional_targets` mechanism.

    Returns per-sample dict:
      rgb   : float32 tensor [3, H, W]  ImageNet-normalised
      irt   : float32 tensor [1, H, W]  min-max normalised to [0, 1]
      mask  : int64 tensor  [H, W]
      image_id : int
    """

    def __init__(
        self,
        annotation_file: str,
        rgb_dir: str,
        irt_dir: str,
        image_size: int = 512,
        augment: bool = True,
    ):
        self.coco      = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.rgb_dir   = Path(rgb_dir)
        self.irt_dir   = Path(irt_dir)
        self.image_size = image_size

        # Build the spatial+photometric pipeline and add 'irt' as a second
        # image target so all spatial transforms are shared with the mask.
        base_pipeline = _build_augmentation_pipeline(image_size, augment)
        self.transform = A.Compose(
            base_pipeline.transforms,
            additional_targets={"irt": "image"},
        )

        logger.info(
            "MultiModalDataset | %d images | augment=%s | RGB=%s IRT=%s",
            len(self.image_ids), augment, rgb_dir, irt_dir,
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id   = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        stem     = Path(img_info["file_name"]).stem   # '000000'

        # ── Load images ────────────────────────────────────────────────────
        rgb_path = _find_image(self.rgb_dir, stem)
        irt_path = _find_image(self.irt_dir, stem)

        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            raise FileNotFoundError(f"Cannot read RGB: {rgb_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)   # uint8 HWC

        irt_gray = cv2.imread(str(irt_path), cv2.IMREAD_GRAYSCALE)
        if irt_gray is None:
            raise FileNotFoundError(f"Cannot read IRT: {irt_path}")
        # albumentations expects HWC; stack to 3-ch then revert after transform
        irt_3ch = np.stack([irt_gray, irt_gray, irt_gray], axis=-1)

        # ── Build segmentation mask ─────────────────────────────────────────
        mask = _build_mask(
            self.coco, img_id, img_info["height"], img_info["width"]
        )

        # ── Apply synchronised transforms ───────────────────────────────────
        out      = self.transform(image=rgb, irt=irt_3ch, mask=mask)
        rgb_aug  = out["image"]           # uint8 HWC [H, W, 3]
        irt_aug  = out["irt"][:, :, 0]   # uint8 HW  (all 3 ch identical)
        mask_aug = out["mask"]            # uint8 HW

        # ── Normalise & convert to tensors ──────────────────────────────────
        rgb_t  = torch.from_numpy(
            _normalize_rgb(rgb_aug).transpose(2, 0, 1)
        ).float()                                              # [3, H, W]

        irt_t  = torch.from_numpy(
            _normalize_irt(irt_aug)[np.newaxis, ...]
        ).float()                                              # [1, H, W]

        mask_t = torch.from_numpy(mask_aug.astype(np.int64)) # [H, W]

        return {"rgb": rgb_t, "irt": irt_t, "mask": mask_t, "image_id": img_id}


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------

class RandomGaussianNoise:
    """Retained for import compatibility. Noise is now applied via albumentations."""

    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std
