

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

from dataset import COCOSegmentationDataset
from CNNVIT import (
    MobileViTSegmentationHead,
    DefectDetectionTrainer,
    _log_final_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Combined CE + Dice loss
# ---------------------------------------------------------------------------

class _CombinedCEDiceLoss(nn.Module):
    """
    Weighted CrossEntropy + soft Dice loss.

    Dice is computed over defect classes only (skips background=0) so the
    loss directly optimises the per-class IoU / Dice metrics we report.
    ce_weight + dice_weight should sum to 1.0.
    """

    def __init__(
        self,
        class_weights: torch.Tensor,
        num_classes: int,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.ce        = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        self.ce_w      = ce_weight
        self.dice_w    = dice_weight
        self.smooth    = smooth
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)

        probs = torch.softmax(logits, dim=1)          # [B, C, H, W]
        valid = (targets >= 0) & (targets < self.num_classes)

        dice_sum = torch.tensor(0.0, device=logits.device)
        n_defect = self.num_classes - 1               # skip background
        for c in range(1, self.num_classes):
            p = probs[:, c][valid]
            t = (targets[valid] == c).float()
            dice_sum = dice_sum + (
                1.0 - (2.0 * (p * t).sum() + self.smooth)
                    / (p.sum() + t.sum() + self.smooth)
            )

        return self.ce_w * ce_loss + self.dice_w * (dice_sum / n_defect)


# ---------------------------------------------------------------------------
# Data-driven class weight computation
# ---------------------------------------------------------------------------

def _compute_class_weights(annotation_file: str, num_classes: int) -> list[float]:
    """
    Derive inverse-frequency class weights from rendered annotation masks.

    Pixels not covered by any annotation count as background (class 0).
    Weights are clipped to [0.1, 10] and normalised so their mean = 1.
    """
    coco         = COCO(annotation_file)
    pixel_counts = np.zeros(num_classes, dtype=np.int64)

    for img_id, img_info in coco.imgs.items():
        h = int(img_info.get("height") or 1)
        w = int(img_info.get("width")  or 1)
        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            cat_id = int(ann["category_id"])
            if 0 < cat_id < num_classes:
                for seg in ann.get("segmentation", []):
                    pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                    if len(pts) >= 3:
                        cv2.fillPoly(mask, [pts], cat_id)

        for c in range(num_classes):
            pixel_counts[c] += int((mask == c).sum())

    total   = pixel_counts.sum()
    weights = total / (num_classes * pixel_counts.clip(min=1).astype(np.float64))
    weights = np.clip(weights, 0.1, 10.0)
    weights /= weights.mean()                         # normalise mean → 1

    logger.info(
        "Class weights (data-driven): %s",
        {i: f"{w:.3f}" for i, w in enumerate(weights)},
    )
    return weights.tolist()


# ---------------------------------------------------------------------------
# Single-modal segmentation model  (MobileViTv2)
# ---------------------------------------------------------------------------

class MobileViTv2SegmentationModel(nn.Module):
    """MobileViTv2-1.0 backbone + 5-stage upsampling head for single-modal segmentation."""

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        hidden_dim: int = 256,
        class_weights: list | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes

        try:
            self.backbone = timm.create_model("mobilevitv2_100", pretrained=pretrained)
            logger.info("Backbone loaded: mobilevitv2_100")
        except Exception:
            logger.warning("mobilevitv2_100 unavailable — falling back to mobilevitv2_075")
            self.backbone = timm.create_model("mobilevitv2_075", pretrained=pretrained)

        backbone_ch = getattr(self.backbone, "num_features", 256)

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
        return self.seg_head(self._extract_features(x))


# ---------------------------------------------------------------------------
# Early Fusion model  (4-channel RGBT → MobileViTv2 backbone)
# ---------------------------------------------------------------------------

class EarlyFusionSegmentationModelV2(nn.Module):
    """
    Early Fusion segmentation model using MobileViTv2-100.

    A learnable 1×1 channel-projection layer maps the 4-channel fused TIFF
    (RGB + IR) down to 3 channels before the pretrained MobileViTv2 backbone.

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
            logger.info("Backbone loaded: mobilevitv2_100 (early fusion)")
        except Exception:
            logger.warning("mobilevitv2_100 unavailable — falling back to mobilevitv2_075")
            self.backbone = timm.create_model("mobilevitv2_075", pretrained=pretrained)

        backbone_ch = getattr(self.backbone, "num_features", 256)
        self.seg_head = MobileViTSegmentationHead(backbone_ch, num_classes, hidden_dim)

        if class_weights is None:
            class_weights = [0.2, 3.0, 2.0, 1.0, 3.0]
        weights = torch.tensor(class_weights[:num_classes], dtype=torch.float32)
        self.criterion = _CombinedCEDiceLoss(weights, num_classes)

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


# ---------------------------------------------------------------------------
# Single-modal pipeline  (fused images, MobileViTv2)
# ---------------------------------------------------------------------------

class DefectDetectionPipelineV2:
    """End-to-end pipeline for the single-modal MobileViTv2 model."""

    def __init__(
        self,
        dataset_root: str,
        num_classes: int = 5,
        batch_size: int = 8,
        num_epochs: int = 10,
        image_size: int = 256,
        checkpoint_dir: str = "./checkpoints",
        lr: float = 1e-3,
    ):
        self.dataset_root   = Path(dataset_root)
        self.num_classes    = num_classes
        self.batch_size     = batch_size
        self.num_epochs     = num_epochs
        self.image_size     = image_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.model   = MobileViTv2SegmentationModel(num_classes=num_classes, pretrained=True)
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.trainer = DefectDetectionTrainer(self.model, device=self.device, lr=lr)
        logger.info("DefectDetectionPipelineV2 ready | device=%s", self.device)

    def train(
        self,
        train_annotation: str = "instances_train.json",
        val_annotation: str   = "instances_val.json",
        early_stop_patience: int = 20,
    ) -> dict:
        train_ds = COCOSegmentationDataset(
            str(self.dataset_root), train_annotation,
            image_size=self.image_size, augment=True,
        )
        val_ds = COCOSegmentationDataset(
            str(self.dataset_root), val_annotation,
            image_size=self.image_size, augment=False,
        )

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)

        logger.info("Train=%d  Val=%d", len(train_ds), len(val_ds))
        history = {"train": [], "val": []}
        no_improve = 0

        for epoch in range(self.num_epochs):
            logger.info("── Epoch %d/%d ──", epoch + 1, self.num_epochs)

            train_m = self.trainer.train_epoch(train_loader, self.num_classes)
            val_m   = self.trainer.validate(val_loader, self.num_classes)
            logger.info("Train: %s", {k: f"{v:.4f}" for k, v in train_m.items()
                                       if not isinstance(v, list)})
            logger.info("Val:   %s", {k: f"{v:.4f}" for k, v in val_m.items()
                                       if not isinstance(v, list)})

            history["train"].append(train_m)
            history["val"].append(val_m)

            score = val_m.get("mean_iou", 0.0)
            self.trainer.scheduler.step(score)

            if score > self.trainer.best_metric:
                self.trainer.best_metric = score
                self.trainer.best_epoch  = epoch
                no_improve = 0
                ckpt = self.checkpoint_dir / f"best_single_modal_v2_epoch{epoch:03d}.pth"
                self.trainer.save_checkpoint(str(ckpt), epoch, val_m)
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        best_val_m = history["val"][self.trainer.best_epoch]
        _log_final_metrics(best_val_m, f"Best Val Metrics  (epoch {self.trainer.best_epoch + 1})")
        return history

    def predict(self, image_path: str, checkpoint_path: Optional[str] = None) -> dict:
        if checkpoint_path:
            self.trainer.load_checkpoint(checkpoint_path)
        self.model.eval()

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        img_r = cv2.resize(img, (self.image_size, self.image_size))
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_t = torch.from_numpy(
            ((img_r.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_t)
            mask   = logits.argmax(dim=1).cpu().numpy()[0]

        mask_orig = cv2.resize(
            mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        return {"mask": mask_orig, "image": img}


# ---------------------------------------------------------------------------
# Early Fusion pipeline  (4-channel fused TIFFs, MobileViTv2)
# ---------------------------------------------------------------------------

class EarlyFusionPipelineV2:
    """
    End-to-end pipeline for the Early Fusion MobileViTv2 model.

    Reads 4-channel fused TIFFs (RGB+IR) from sub-directories of fused_dir.
    The COCO JSON is looked up in annotation_dir, not inside fused_dir.
    """

    def __init__(
        self,
        fused_dir: str,
        annotation_dir: str,
        num_classes: int = 5,
        batch_size: int = 4,
        num_epochs: int = 20,
        image_size: int = 256,
        checkpoint_dir: str = "./checkpoints",
        lr: float = 1e-3,
        hidden_dim: int = 256,
        train_annotation_file: Optional[str] = None,
        warmup_epochs: int = 10,
    ):
        self.fused_dir      = Path(fused_dir)
        self.annotation_dir = Path(annotation_dir)
        self.num_classes    = num_classes
        self.batch_size     = batch_size
        self.num_epochs     = num_epochs
        self.image_size     = image_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # ── Data-driven class weights ────────────────────────────────────────
        if train_annotation_file and Path(train_annotation_file).exists():
            class_weights = _compute_class_weights(train_annotation_file, num_classes)
        else:
            logger.warning("train_annotation_file not provided — using default class weights")
            class_weights = [0.2, 3.0, 2.0, 1.0, 3.0]

        self.model  = EarlyFusionSegmentationModelV2(
            num_classes=num_classes,
            pretrained=True,
            hidden_dim=hidden_dim,
            in_channels=4,
            class_weights=class_weights,
        )
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.trainer = DefectDetectionTrainer(self.model, device=self.device, lr=lr)

        # ── Replace ReduceLROnPlateau with linear warmup → cosine decay ──────
        warmup_epochs = min(warmup_epochs, num_epochs // 10)
        cosine_epochs = max(num_epochs - warmup_epochs, 1)
        warmup_sched  = optim.lr_scheduler.LinearLR(
            self.trainer.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_sched  = optim.lr_scheduler.CosineAnnealingLR(
            self.trainer.optimizer,
            T_max=cosine_epochs,
            eta_min=1e-6,
        )
        self.trainer.scheduler = optim.lr_scheduler.SequentialLR(
            self.trainer.optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
        logger.info(
            "EarlyFusionPipelineV2 ready | device=%s | warmup=%d epochs | cosine=%d epochs",
            self.device, warmup_epochs, cosine_epochs,
        )

    def train(
        self,
        train_annotation: str = "instances_train.json",
        val_annotation: str   = "instances_val.json",
        early_stop_patience: int = 20,
    ) -> dict:
        train_ds = COCOSegmentationDataset(
            root_dir=str(self.fused_dir),
            annotation_file=str(self.annotation_dir / train_annotation),
            image_size=self.image_size,
            augment=True,
        )
        val_ds = COCOSegmentationDataset(
            root_dir=str(self.fused_dir),
            annotation_file=str(self.annotation_dir / val_annotation),
            image_size=self.image_size,
            augment=False,
        )

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)

        logger.info("Train=%d  Val=%d", len(train_ds), len(val_ds))
        history    = {"train": [], "val": []}
        no_improve = 0

        for epoch in range(self.num_epochs):
            logger.info("── Epoch %d/%d ──", epoch + 1, self.num_epochs)

            train_m = self.trainer.train_epoch(train_loader, self.num_classes)
            val_m   = self.trainer.validate(val_loader, self.num_classes)

            logger.info("Train: %s", {k: f"{v:.4f}" for k, v in train_m.items()
                                       if not isinstance(v, list)})
            logger.info("Val:   %s", {k: f"{v:.4f}" for k, v in val_m.items()
                                       if not isinstance(v, list)})

            history["train"].append(train_m)
            history["val"].append(val_m)

            score = val_m.get("mean_iou", 0.0)
            self.trainer.scheduler.step()   # cosine/sequential — no arg needed

            if score > self.trainer.best_metric:
                self.trainer.best_metric = score
                self.trainer.best_epoch  = epoch
                no_improve = 0
                ckpt = self.checkpoint_dir / f"best_early_fusion_v2_epoch{epoch:03d}.pth"
                self.trainer.save_checkpoint(str(ckpt), epoch, val_m)
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        best_val_m = history["val"][self.trainer.best_epoch]
        _log_final_metrics(best_val_m, f"Best Val Metrics  (epoch {self.trainer.best_epoch + 1})")
        return history

    def predict(self, fused_path: str, checkpoint_path: Optional[str] = None) -> dict:
        """Run inference on a single 4-channel fused TIFF."""
        if checkpoint_path:
            self.trainer.load_checkpoint(checkpoint_path)
        self.model.eval()

        raw = cv2.imread(fused_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Cannot read: {fused_path}")
        orig_h, orig_w = raw.shape[:2]

        bgr = raw[:, :, :3]
        ir  = raw[:, :, 3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        rgb_r = cv2.resize(rgb, (self.image_size, self.image_size))
        rgb_n = (rgb_r.astype(np.float32) / 255.0 - mean) / std

        ir_r  = cv2.resize(ir, (self.image_size, self.image_size)).astype(np.float32)
        lo, hi = ir_r.min(), ir_r.max()
        ir_n  = (ir_r - lo) / (hi - lo + 1e-7)

        fused_n = np.concatenate(
            [rgb_n, ir_n[..., np.newaxis]], axis=2
        ).transpose(2, 0, 1)  # [4, H, W]
        img_t = torch.from_numpy(fused_n).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(img_t)
            mask   = logits.argmax(dim=1).cpu().numpy()[0]

        mask_orig = cv2.resize(
            mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        return {"mask": mask_orig, "rgb": rgb}


# ---------------------------------------------------------------------------
# Entry point — runs Early Fusion v2 training from config.yaml
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import os
    import yaml
    from dataset_utils import CVATXMLToCOCOConverter

    with open("./config.yaml") as f:
        cfg = yaml.safe_load(f)

    train_json = cfg["dataset"]["train_annotation_file"]
    val_json   = cfg["dataset"]["val_annotation_file"]

    regen = True
    if os.path.exists(train_json):
        with open(train_json) as f:
            regen = len(json.load(f).get("annotations", [])) == 0
    if regen:
        logger.info("Regenerating COCO annotations from CVAT XML…")
        converter = CVATXMLToCOCOConverter(cfg["dataset"]["xml_annotation"])
        converter.convert(
            output_train=train_json,
            output_val=val_json,
            train_ratio=cfg["dataset"].get("train_ratio", 0.8),
        )
    else:
        logger.info("COCO annotations already exist — skipping conversion.")

    pipeline = EarlyFusionPipelineV2(
        fused_dir              = cfg["dataset"]["fused_dir"],
        annotation_dir         = cfg["dataset"]["annotation_dir"],
        num_classes            = cfg["dataset"]["num_classes"],
        batch_size             = cfg["training"]["batch_size"],
        num_epochs             = cfg["training"]["num_epochs"],
        image_size             = cfg["dataset"]["image_size"],
        checkpoint_dir         = cfg["checkpoint"]["save_dir"],
        lr                     = cfg["training"]["learning_rate"],
        train_annotation_file  = train_json,
    )

    logger.info("Starting Early Fusion v2 training…")
    history = pipeline.train(
        train_annotation    = Path(train_json).name,
        val_annotation      = Path(val_json).name,
        early_stop_patience = cfg["training"].get("early_stopping_patience", 20),
    )
    logger.info("Training complete. Best val IoU: %.4f", pipeline.trainer.best_metric)
