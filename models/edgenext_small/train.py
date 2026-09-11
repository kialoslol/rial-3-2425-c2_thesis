"""
End-to-end training/inference pipeline for the EdgeNeXt-Small early-fusion
model.

Uses the same training recipe as models/mobilenet_v4, models/early_fusion_v2
and models/CNN_VIT_SELF (all take the same 4-channel fused RGBT input):
data-driven class weights, moisture oversampling, cosine annealing with warm
restarts, and partial-mIoU early stopping — so the only real difference
between these models is the backbone.

Run directly: python models/edgenext_small/train.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from common.dataset import COCOSegmentationDataset
from common.dataset_utils import compute_class_weights, build_moisture_sampler
from common.trainer import DefectDetectionTrainer, log_final_metrics, save_and_prune_checkpoint
from models.edgenext_small.model import EdgeNeXtSegmentationModel

logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).resolve().parent


class EdgeNeXtPipeline:
    """End-to-end pipeline for the EdgeNeXt-Small early-fusion model."""

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
        train_annotation_file: Optional[str] = None,
        moisture_confidence_threshold: float = 0.6,
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
        self.moisture_threshold = moisture_confidence_threshold
        self.keep_best_n         = keep_best_n
        self.checkpoint_history: list = []

        # -- Data-driven class weights ----------------------------------------
        if train_annotation_file and Path(train_annotation_file).exists():
            class_weights = compute_class_weights(train_annotation_file, num_classes)
        else:
            logger.warning("train_annotation_file not provided — using default class weights")
            class_weights = [0.5, 3.0, 2.0, 1.0, 10.0]

        self.model = EdgeNeXtSegmentationModel(
            num_classes=num_classes,
            pretrained=True,
            hidden_dim=hidden_dim,
            in_channels=4,
            class_weights=class_weights,
        )
        self.device  = "cuda" if torch.cuda.is_available() else "cpu"
        self.trainer = DefectDetectionTrainer(self.model, device=self.device, lr=lr)

        # -- Cosine annealing with warm restarts ------------------------------
        self.trainer.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.trainer.optimizer,
            T_0=min(50, max(num_epochs // 6, 10)),
            T_mult=2,
            eta_min=1e-6,
        )
        logger.info(
            "EdgeNeXtPipeline ready | device=%s | scheduler=CosineWarmRestarts "
            "T_0=%d T_mult=2 | moisture_threshold=%.2f",
            self.device,
            min(50, max(num_epochs // 6, 10)),
            moisture_confidence_threshold,
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

        # -- Oversample moisture images so each epoch sees more moisture ------
        sampler = build_moisture_sampler(train_ds.coco, train_ds.image_ids,
                                          oversample_factor=4.0)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  sampler=sampler, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)

        logger.info("Train=%d  Val=%d", len(train_ds), len(val_ds))
        history    = {"train": [], "val": []}
        no_improve = 0

        try:
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

                self.trainer.scheduler.step()

                # -- Partial mIoU for early stopping (exclude moisture) -------
                iou_list = val_m.get("iou_per_class", [])
                partial  = [iou_list[c] for c in (0, 1, 2, 3) if c < len(iou_list)]
                score    = float(np.mean(partial)) if partial else val_m.get("mean_iou", 0.0)

                if score > self.trainer.best_metric:
                    self.trainer.best_metric = score
                    self.trainer.best_epoch  = epoch
                    no_improve = 0
                    save_and_prune_checkpoint(
                        self.trainer, self.checkpoint_dir, self.checkpoint_history,
                        "best_edgenext", epoch, val_m, score, self.keep_best_n,
                    )
                else:
                    no_improve += 1
                    if no_improve >= early_stop_patience:
                        logger.info("Early stopping at epoch %d", epoch + 1)
                        break

        except KeyboardInterrupt:
            logger.info("\n[INTERRUPTED] Ctrl+C received — stopping training early.")

        if not history["val"]:
            logger.warning("No epochs completed — nothing to report.")
            return history

        best_epoch = self.trainer.best_epoch if self.trainer.best_metric > 0.0 else 0
        best_ckpt  = self.checkpoint_dir / f"best_edgenext_epoch{best_epoch:03d}.pth"
        if best_ckpt.exists():
            logger.info("Loading best checkpoint: %s", best_ckpt)
            self.trainer.load_checkpoint(str(best_ckpt))
        else:
            logger.info("Best checkpoint not found on disk — reporting in-memory metrics.")

        best_val_m = history["val"][best_epoch]
        log_final_metrics(best_val_m, f"Best Val Metrics  (epoch {best_epoch + 1})")
        return history

    def predict(self, fused_path: str, checkpoint_path: Optional[str] = None,
                moisture_threshold: Optional[float] = None) -> dict:
        """
        Run inference on a single 4-channel fused TIFF.

        moisture_threshold: minimum softmax probability to accept a moisture
        prediction. Defaults to self.moisture_threshold (0.6). Set to None to
        disable.
        """
        if checkpoint_path:
            self.trainer.load_checkpoint(checkpoint_path)
        self.model.eval()

        thr = moisture_threshold if moisture_threshold is not None else self.moisture_threshold

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
        ).transpose(2, 0, 1)
        img_t = torch.from_numpy(fused_n).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.model(img_t)
            probs  = torch.softmax(logits, dim=1)
            mask   = logits.argmax(dim=1).cpu().numpy()[0]

            if thr is not None and thr > 0.0:
                moisture_prob = probs[0, 4].cpu().numpy()
                low_conf_moisture = (mask == 4) & (moisture_prob < thr)
                second_best = logits[0].clone()
                second_best[4] = -1e9
                fallback = second_best.argmax(dim=0).cpu().numpy()
                mask[low_conf_moisture] = fallback[low_conf_moisture]

        mask_orig = cv2.resize(
            mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
        return {"mask": mask_orig, "rgb": rgb}


if __name__ == "__main__":
    import json
    import os
    import yaml
    from common.dataset_utils import CVATXMLToCOCOConverter

    with open(REPO_ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    train_json = cfg["dataset"]["train_annotation_file"]
    val_json   = cfg["dataset"]["val_annotation_file"]

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

    checkpoint_dir = str(THIS_DIR / "checkpoints")

    pipeline = EdgeNeXtPipeline(
        fused_dir              = cfg["dataset"]["fused_dir"],
        annotation_dir         = cfg["dataset"]["annotation_dir"],
        num_classes            = cfg["dataset"]["num_classes"],
        batch_size             = cfg["training"]["batch_size"],
        num_epochs             = cfg["training"]["num_epochs"],
        image_size             = cfg["dataset"]["image_size"],
        checkpoint_dir         = checkpoint_dir,
        lr                     = cfg["training"]["learning_rate"],
        train_annotation_file  = train_json,
        keep_best_n            = cfg["checkpoint"].get("keep_best_n", 10),
    )

    logger.info("Starting EdgeNeXt-Small training...")
    history = pipeline.train(
        train_annotation    = Path(train_json).name,
        val_annotation      = Path(val_json).name,
        early_stop_patience = cfg["training"].get("early_stopping_patience", 20),
    )
    logger.info("Training complete. Best partial-mIoU: %.4f", pipeline.trainer.best_metric)

    history_path = Path(checkpoint_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f)
    logger.info("Training history saved: %s", history_path)
