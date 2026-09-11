"""
End-to-end training/inference pipeline for the Early Fusion v1 model.

Reads 4-channel fused TIFFs (RGB+IR) produced by scripts/channelstack.py.
Images live in sub-directories of fused_dir (e.g. fuse_Binondo/, FUSED/,
Pasay-LP/, Binondocropped/); the file_name in the COCO JSON already
encodes the sub-directory (e.g. 'fuse_Binondo/000000.tiff').

Run directly: python models/early_fusion_v1/train.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.dataset import COCOSegmentationDataset
from common.trainer import DefectDetectionTrainer, log_final_metrics, save_and_prune_checkpoint
from models.early_fusion_v1.model import EarlyFusionSegmentationModel

logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent


class EarlyFusionPipeline:
    """
    End-to-end pipeline for the Early Fusion MobileViT model.

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
        checkpoint_dir: str = str(THIS_DIR / "checkpoints"),
        lr: float = 1e-3,
        hidden_dim: int = 256,
        keep_best_n: int = 10,
    ):
        self.fused_dir      = Path(fused_dir)
        self.annotation_dir = Path(annotation_dir)
        self.num_classes    = num_classes
        self.batch_size     = batch_size
        self.num_epochs     = num_epochs
        self.image_size     = image_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_n       = keep_best_n
        self.checkpoint_history: list = []

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
            logger.info("-- Epoch %d/%d --", epoch + 1, self.num_epochs)

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
                save_and_prune_checkpoint(
                    self.trainer, self.checkpoint_dir, self.checkpoint_history,
                    "best_early_fusion", epoch, val_m, score, self.keep_best_n,
                )
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        best_val_m = history["val"][self.trainer.best_epoch]
        log_final_metrics(best_val_m, f"Best Val Metrics  (epoch {self.trainer.best_epoch + 1})")
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


if __name__ == "__main__":
    import yaml
    from common.dataset_utils import CVATXMLToCOCOConverter

    with open(REPO_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    # -- Step 1: convert CVAT annotations to COCO JSON (skip if already done) --
    train_json = cfg["dataset"]["train_annotation_file"]
    val_json   = cfg["dataset"]["val_annotation_file"]

    import json, os
    regen = True
    if os.path.exists(train_json):
        with open(train_json) as f:
            regen = len(json.load(f).get("annotations", [])) == 0
    if regen:
        logger.info("Regenerating COCO annotations from CVAT XML...")
        converter = CVATXMLToCOCOConverter(cfg["dataset"]["xml_annotation"])
        converter.convert(
            output_train=train_json,
            output_val=val_json,
            train_ratio=cfg["dataset"].get("train_ratio", 0.8),
        )
    else:
        logger.info("COCO annotations already exist — skipping conversion.")

    # -- Step 2: train Early Fusion model --------------------------------------
    pipeline = EarlyFusionPipeline(
        fused_dir       = cfg["dataset"]["fused_dir"],
        annotation_dir  = cfg["dataset"]["annotation_dir"],
        num_classes     = cfg["dataset"]["num_classes"],
        batch_size      = cfg["training"]["batch_size"],
        num_epochs      = cfg["training"]["num_epochs"],
        image_size      = cfg["dataset"]["image_size"],
        checkpoint_dir  = str(THIS_DIR / "checkpoints"),
        lr              = cfg["training"]["learning_rate"],
        keep_best_n     = cfg["checkpoint"].get("keep_best_n", 10),
    )

    logger.info("Starting Early Fusion training...")
    history = pipeline.train(
        train_annotation    = Path(train_json).name,
        val_annotation      = Path(val_json).name,
        early_stop_patience = cfg["training"].get("early_stopping_patience", 20),
    )
    logger.info("Training complete. Best val IoU: %.4f", pipeline.trainer.best_metric)
