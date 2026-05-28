"""
MobileViT V3 CNN-ViT Defect Detection
Provides:
  MobileViTSegmentationModel  — single-modal (3-ch image) model
  EarlyFusionSegmentationModel — 4-channel early-fusion model (RGB+IR → one backbone)
  DefectDetectionMetrics      — per-class and aggregate metrics
  DefectDetectionTrainer      — training loop (single- and multi-modal aware)
  DefectDetectionPipeline     — end-to-end pipeline for the single-modal model
  EarlyFusionPipeline         — end-to-end pipeline for Early Fusion (4-ch fused TIFFs)

Class IDs (shared across all modules):
  0  background
  1  crack
  2  spall
  3  delamination  (reserved; no annotations yet)
  4  moisture
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import COCOSegmentationDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-modal segmentation model
# ---------------------------------------------------------------------------

class MobileViTSegmentationHead(nn.Module):
    """
    Decoder for the single-modal model.

    5 × 2× bilinear upsample stages → 32× total upsampling.
    With MobileViT-S (stride 32) and 512×512 input the backbone outputs
    [B, 640, 16, 16]; the head returns [B, num_classes, 512, 512].

    Channel schedule  (hidden_dim=256):
      640 → 256 → 128 → 64 → 64 → 64 → num_classes
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        d  = hidden_dim
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
            _block(in_channels, d),   # 16 →  32
            _block(d,           d2),  # 32 →  64
            _block(d2,          d4),  # 64 → 128
            _block(d4,          d4),  # 128 → 256
            _block(d4,          d4),  # 256 → 512
        )
        self.seg_head = nn.Conv2d(d4, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seg_head(self.decoder(x))


class MobileViTSegmentationModel(nn.Module):
    """MobileViT-S backbone + 5-stage upsampling head for single-modal segmentation."""

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
            self.backbone = timm.create_model("mobilevit_s", pretrained=pretrained)
            logger.info("Backbone loaded: mobilevit_s")
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
        return self.seg_head(self._extract_features(x))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class DefectDetectionMetrics:
    """Per-class IoU, Dice, pixel accuracy, and macro-F1."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            predictions : [B, C, H, W] logits
            targets     : [B, H, W]    ground-truth class indices
        """
        pred_cls = predictions.argmax(dim=1).cpu().numpy().ravel()
        gt_cls   = targets.cpu().numpy().ravel()

        valid = gt_cls < self.num_classes
        pred_cls, gt_cls = pred_cls[valid], gt_cls[valid]

        for gt, pr in zip(gt_cls, pred_cls):
            self.confusion_matrix[gt, pr] += 1

    def compute(self) -> Dict[str, float]:
        cm = self.confusion_matrix.astype(np.float64)

        # Per-class IoU
        iou_per_class = []
        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            denom = tp + fp + fn
            iou_per_class.append(tp / denom if denom > 0 else 0.0)

        # Per-class Dice
        dice_per_class = []
        for c in range(self.num_classes):
            tp = cm[c, c]
            denom = cm[c, :].sum() + cm[:, c].sum()
            dice_per_class.append(2 * tp / denom if denom > 0 else 0.0)

        # Pixel accuracy
        accuracy = np.diag(cm).sum() / (cm.sum() + 1e-7)

        # Per-class precision, recall, F1 (all classes including background)
        precision_per_class = []
        recall_per_class    = []
        f1_per_class        = []
        for c in range(self.num_classes):
            tp   = cm[c, c]
            fp   = cm[:, c].sum() - tp
            fn   = cm[c, :].sum() - tp
            prec = float(tp / (tp + fp + 1e-7))
            rec  = float(tp / (tp + fn + 1e-7))
            f1   = float(2 * prec * rec / (prec + rec + 1e-7))
            precision_per_class.append(prec)
            recall_per_class.append(rec)
            f1_per_class.append(f1)

        # Aggregate over defect classes only (skip background=0)
        defect_prec = precision_per_class[1:]
        defect_f1   = f1_per_class[1:]

        return {
            "accuracy":           float(accuracy),
            "mean_iou":           float(np.mean(iou_per_class)),
            "mean_dice":          float(np.mean(dice_per_class)),
            "macro_f1":           float(np.mean(defect_f1))   if defect_f1   else 0.0,
            "mAP":                float(np.mean(defect_prec)) if defect_prec else 0.0,
            "iou_per_class":      iou_per_class,
            "dice_per_class":     dice_per_class,
            "precision_per_class": precision_per_class,
            "recall_per_class":   recall_per_class,
            "f1_per_class":       f1_per_class,
        }

    # Legacy single-call interface (used by progress bars)
    def compute_metrics(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        self.reset()
        self.update(predictions, targets)
        result = self.compute()
        return {
            "accuracy":  result["accuracy"],
            "iou":       result["mean_iou"],
            "dice":      result["mean_dice"],
            "f1_score":  result["macro_f1"],
        }


# ---------------------------------------------------------------------------
# Metric display helper
# ---------------------------------------------------------------------------

_CLASS_NAMES = ["background", "crack", "spall", "delamination", "moisture"]

def _log_final_metrics(metrics: dict, label: str = "Final") -> None:
    """Print a formatted summary of averaged and per-class metrics."""
    sep = "─" * 72
    logger.info(sep)
    logger.info("  %s", label)
    logger.info(sep)
    logger.info("  %-16s %.4f", "Loss:",      metrics.get("loss",      float("nan")))
    logger.info("  %-16s %.4f", "Accuracy:",  metrics.get("accuracy",  0.0))
    logger.info("  %-16s %.4f", "Mean IoU:",  metrics.get("mean_iou",  0.0))
    logger.info("  %-16s %.4f", "Mean Dice:", metrics.get("mean_dice", 0.0))
    logger.info("  %-16s %.4f", "Macro F1:",  metrics.get("macro_f1",  0.0))
    logger.info("  %-16s %.4f", "mAP:",       metrics.get("mAP",       0.0))
    logger.info(sep)
    logger.info("  %-14s  %7s  %7s  %9s  %7s  %7s",
                "Class", "IoU", "Dice", "Precision", "Recall", "F1")
    logger.info("  %-14s  %7s  %7s  %9s  %7s  %7s",
                "─" * 14, "─" * 7, "─" * 7, "─" * 9, "─" * 7, "─" * 7)
    iou_list   = metrics.get("iou_per_class",       [])
    dice_list  = metrics.get("dice_per_class",      [])
    prec_list  = metrics.get("precision_per_class", [])
    rec_list   = metrics.get("recall_per_class",    [])
    f1_list    = metrics.get("f1_per_class",        [])
    for i, vals in enumerate(zip(iou_list, dice_list, prec_list, rec_list, f1_list)):
        iou, dice, prec, rec, f1 = vals
        name = _CLASS_NAMES[i] if i < len(_CLASS_NAMES) else f"class_{i}"
        logger.info("  %-14s  %7.4f  %7.4f  %9.4f  %7.4f  %7.4f",
                    name, iou, dice, prec, rec, f1)
    logger.info(sep)


# ---------------------------------------------------------------------------
# Trainer  (single- and multi-modal aware)
# ---------------------------------------------------------------------------

class DefectDetectionTrainer:
    """
    Generic training loop.

    Detects batch type automatically:
      - single-modal batches contain key 'image'  → model(image)
      - multi-modal batches contain keys 'rgb'/'irt' → model(rgb, irt)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.device    = device
        self.model     = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=10
        )
        self.best_metric = 0.0
        self.best_epoch  = 0
        logger.info("Trainer initialised | device=%s | lr=%g", device, lr)

    # ------------------------------------------------------------------

    def _forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the model on a batch and return (logits, masks)."""
        if "rgb" in batch:
            rgb  = batch["rgb"].to(self.device)
            irt  = batch["irt"].to(self.device)
            mask = batch["mask"].to(self.device)
            return self.model(rgb, irt), mask
        else:
            img  = batch["image"].to(self.device)
            mask = batch["mask"].to(self.device)
            return self.model(img), mask

    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader, num_classes: int) -> Dict[str, float]:
        self.model.train()
        metrics    = DefectDetectionMetrics(num_classes)
        total_loss = 0.0
        bar        = tqdm(loader, desc="Train")

        for batch in bar:
            self.optimizer.zero_grad()
            logits, masks = self._forward(batch)
            loss = self.model.criterion(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            metrics.update(logits.detach(), masks)

            bar.set_postfix(loss=f"{loss.item():.4f}")

        result = metrics.compute()
        result["loss"] = total_loss / len(loader)
        return result

    def validate(self, loader: DataLoader, num_classes: int) -> Dict[str, float]:
        self.model.eval()
        metrics    = DefectDetectionMetrics(num_classes)
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(loader, desc="Val"):
                logits, masks = self._forward(batch)
                loss = self.model.criterion(logits, masks)
                total_loss += loss.item()
                metrics.update(logits, masks)

        result = metrics.compute()
        result["loss"] = total_loss / len(loader)
        return result

    # ------------------------------------------------------------------

    def save_checkpoint(self, filepath: str, epoch: int, metrics: dict):
        torch.save(
            {
                "epoch":           epoch,
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "metrics":         metrics,
                "best_metric":     self.best_metric,
            },
            filepath,
        )
        logger.info("Checkpoint saved: %s", filepath)

    def load_checkpoint(self, filepath: str) -> dict:
        ckpt = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_metric = ckpt.get("best_metric", 0.0)
        self.best_epoch  = ckpt.get("epoch", 0)
        logger.info("Checkpoint loaded: %s", filepath)
        return ckpt


# ---------------------------------------------------------------------------
# Early Fusion model  (4-channel RGBT → single backbone)
# ---------------------------------------------------------------------------

class EarlyFusionSegmentationModel(nn.Module):
    """
    Early Fusion segmentation model.

    A learnable 1×1 channel-projection layer maps the 4-channel fused TIFF
    (RGB + IR) down to 3 channels, which are then fed into the pretrained
    MobileViT-S backbone.  This keeps the backbone weights intact while
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

        # Channel projection: [B, in_channels, H, W] → [B, 3, H, W]
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


# ---------------------------------------------------------------------------
# Single-modal pipeline  (fused images)
# ---------------------------------------------------------------------------

class DefectDetectionPipeline:
    """End-to-end pipeline for the single-modal MobileViT model."""

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

        self.model   = MobileViTSegmentationModel(num_classes=num_classes, pretrained=True)
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.trainer = DefectDetectionTrainer(self.model, device=self.device, lr=lr)
        logger.info("DefectDetectionPipeline ready | device=%s", self.device)

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
                ckpt = self.checkpoint_dir / f"best_single_modal_epoch{epoch:03d}.pth"
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
# Early Fusion pipeline  (4-channel fused TIFFs from ANNOTATED_DATA)
# ---------------------------------------------------------------------------

class EarlyFusionPipeline:
    """
    End-to-end pipeline for the Early Fusion MobileViT model.

    Reads 4-channel fused TIFFs (RGB+IR) produced by channelstack.py.
    Images live in sub-directories of fused_dir (e.g. fuse_Binondo/, FUSED/,
    Pasay-LP/, Binondocropped/); the file_name in the COCO JSON already
    encodes the sub-directory (e.g. 'fuse_Binondo/000000.tiff').

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
    ):
        self.fused_dir      = Path(fused_dir)
        self.annotation_dir = Path(annotation_dir)
        self.num_classes    = num_classes
        self.batch_size     = batch_size
        self.num_epochs     = num_epochs
        self.image_size     = image_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.model  = EarlyFusionSegmentationModel(
            num_classes=num_classes,
            pretrained=True,
            hidden_dim=hidden_dim,
            in_channels=4,
        )
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.trainer = DefectDetectionTrainer(self.model, device=self.device, lr=lr)
        logger.info("EarlyFusionPipeline ready | device=%s", self.device)

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
            self.trainer.scheduler.step(score)

            if score > self.trainer.best_metric:
                self.trainer.best_metric = score
                self.trainer.best_epoch  = epoch
                no_improve = 0
                ckpt = self.checkpoint_dir / f"best_early_fusion_epoch{epoch:03d}.pth"
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
# Entry point — runs Late Fusion training from config.yaml
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml
    from dataset_utils import CVATXMLToCOCOConverter

    with open("./config.yaml") as f:
        cfg = yaml.safe_load(f)

    # ── Step 1: convert CVAT annotations to COCO JSON (skip if already done) ──
    train_json = cfg["dataset"]["train_annotation_file"]
    val_json   = cfg["dataset"]["val_annotation_file"]

    import json, os
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

    # ── Step 2: train Early Fusion model ─────────────────────────────────────
    pipeline = EarlyFusionPipeline(
        fused_dir       = cfg["dataset"]["fused_dir"],
        annotation_dir  = cfg["dataset"]["annotation_dir"],
        num_classes     = cfg["dataset"]["num_classes"],
        batch_size      = cfg["training"]["batch_size"],
        num_epochs      = cfg["training"]["num_epochs"],
        image_size      = cfg["dataset"]["image_size"],
        checkpoint_dir  = cfg["checkpoint"]["save_dir"],
        lr              = cfg["training"]["learning_rate"],
    )

    logger.info("Starting Early Fusion training…")
    history = pipeline.train(
        train_annotation    = Path(train_json).name,
        val_annotation      = Path(val_json).name,
        early_stop_patience = cfg["training"].get("early_stopping_patience", 20),
    )
    logger.info("Training complete. Best val IoU: %.4f", pipeline.trainer.best_metric)
