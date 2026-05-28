"""
Dataset utilities for converting various annotation formats to COCO JSON.

Classes:
  CVATXMLToCOCOConverter  — Convert CVAT polyline/polygon XML to COCO segmentation JSON (primary)
  XMLToCOCOConverter      — Legacy Pascal VOC bounding-box converter (retained for reference)
  MultimodalImageFusion   — Fuse RGB + Thermal images into 4-channel arrays
  DatasetOrganizer        — Split a flat image directory into train/val/test
"""

import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary converter: CVAT XML → COCO JSON (segmentation)
# ---------------------------------------------------------------------------

class CVATXMLToCOCOConverter:
    """
    Convert CVAT 1.1 XML annotations (polyline / polygon shapes) to COCO JSON
    segmentation format.

    Defect class IDs (consistent across all models and datasets):
      0  background
      1  crack        — polyline → dilated thick-line mask (CRACK_THICKNESS px)
      2  spall        — polyline → closed polygon, filled
      3  delamination — polyline/polygon → filled (reserved; no annotations yet)
      4  moisture     — polyline → closed polygon, filled

    Cracks are kept as thick lines because they are genuinely linear features;
    all other defects are area-type and are rendered by closing the polyline
    into a filled polygon.
    """

    CATEGORIES: List[Dict] = [
        {"id": 0, "name": "background",   "supercategory": "defect"},
        {"id": 1, "name": "crack",        "supercategory": "defect"},
        {"id": 2, "name": "spall",        "supercategory": "defect"},
        {"id": 3, "name": "delamination", "supercategory": "defect"},
        {"id": 4, "name": "moisture",     "supercategory": "defect"},
    ]

    # Map label name → category_id (excludes background)
    LABEL_TO_ID: Dict[str, int] = {c["name"]: c["id"] for c in CATEGORIES if c["id"] > 0}

    # Crack polyline thickness in *original image* pixels.
    # Scales down proportionally when the image is resized during training.
    CRACK_THICKNESS: int = 10

    def __init__(self, xml_path: str):
        self.xml_path = Path(xml_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        output_train: str,
        output_val: str,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> Tuple[Dict, Dict]:
        """
        Parse the CVAT XML and write COCO JSON files for train and val splits.

        Args:
            output_train: Destination path for instances_train.json
            output_val:   Destination path for instances_val.json
            train_ratio:  Fraction of images used for training (default 0.8)
            seed:         Random seed for reproducible split

        Returns:
            (train_coco, val_coco) dicts
        """
        random.seed(seed)

        tree = ET.parse(str(self.xml_path))
        root = tree.getroot()

        image_elems = list(root.findall("image"))
        random.shuffle(image_elems)

        n_train = max(1, int(len(image_elems) * train_ratio))
        train_elems = image_elems[:n_train]
        val_elems   = image_elems[n_train:]

        train_coco = self._build_coco_json(train_elems, "train")
        val_coco   = self._build_coco_json(val_elems,   "val")

        for out_path, coco in [(output_train, train_coco), (output_val, val_coco)]:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(coco, f)
            logger.info(
                "[CVAT→COCO] %s: %d images, %d annotations",
                out_path,
                len(coco["images"]),
                len(coco["annotations"]),
            )

        return train_coco, val_coco

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_coco_json(self, image_elems: list, split: str) -> Dict:
        coco = {
            "info": {
                "description": f"Defect Segmentation – {split}",
                "version": "1.0",
                "year": 2026,
            },
            "categories": self.CATEGORIES,
            "images": [],
            "annotations": [],
        }

        ann_id = 1
        for img_id, img_elem in enumerate(image_elems, start=1):
            file_name = img_elem.get("name", "")   # e.g. 'fuse_Binondo/000000.tiff'
            w = int(img_elem.get("width",  0))
            h = int(img_elem.get("height", 0))

            coco["images"].append(
                {"id": img_id, "file_name": file_name, "width": w, "height": h}
            )

            for shape in img_elem:
                tag   = shape.tag                       # 'polyline' or 'polygon'
                label = shape.get("label", "").lower()

                if tag not in ("polyline", "polygon") or label not in self.LABEL_TO_ID:
                    continue

                cat_id   = self.LABEL_TO_ID[label]
                pts_str  = shape.get("points", "")
                if not pts_str:
                    continue

                pts = self._parse_points(pts_str)

                mask = (
                    self._polygon_to_mask(pts, w, h)
                    if tag == "polygon"
                    else self._polyline_to_mask(pts, w, h, label)
                )

                if mask.sum() == 0:
                    continue

                segmentation = self._mask_to_coco_segmentation(mask)
                if not segmentation:
                    continue

                # Tight bounding box from non-zero mask pixels
                rows = np.where(np.any(mask, axis=1))[0]
                cols = np.where(np.any(mask, axis=0))[0]
                y0, y1 = int(rows[0]), int(rows[-1])
                x0, x1 = int(cols[0]), int(cols[-1])

                coco["annotations"].append(
                    {
                        "id":           ann_id,
                        "image_id":     img_id,
                        "category_id":  cat_id,
                        "segmentation": segmentation,
                        "bbox":         [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                        "area":         int(mask.sum()),
                        "iscrowd":      0,
                    }
                )
                ann_id += 1

        return coco

    @staticmethod
    def _parse_points(points_str: str) -> np.ndarray:
        """'x1,y1;x2,y2;...' → float32 array (N, 2)"""
        pts = [p.split(",") for p in points_str.strip().split(";")]
        return np.array([[float(x), float(y)] for x, y in pts], dtype=np.float32)

    def _polyline_to_mask(
        self, points: np.ndarray, w: int, h: int, label: str
    ) -> np.ndarray:
        """
        Render a CVAT polyline annotation as a binary mask.

        Cracks are drawn as thick lines (linear features).
        All other labels are rendered as filled polygons (close the open path).
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        pts  = np.round(points).astype(np.int32)

        if label == "crack":
            for i in range(len(pts) - 1):
                cv2.line(mask, tuple(pts[i]), tuple(pts[i + 1]), 1,
                         thickness=self.CRACK_THICKNESS)
        else:
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 1)
            else:
                cv2.line(mask, tuple(pts[0]), tuple(pts[-1]), 1,
                         thickness=self.CRACK_THICKNESS)
        return mask

    @staticmethod
    def _polygon_to_mask(points: np.ndarray, w: int, h: int) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.uint8)
        pts  = np.round(points).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
        return mask

    @staticmethod
    def _mask_to_coco_segmentation(mask: np.ndarray) -> List[List[float]]:
        """Convert a binary mask to a list of COCO polygon segmentation arrays."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        segs = []
        for c in contours:
            if len(c) >= 3:
                flat = c.flatten().tolist()
                if len(flat) >= 6:
                    segs.append([float(v) for v in flat])
        return segs


# ---------------------------------------------------------------------------
# Legacy Pascal VOC XML → COCO converter (bounding boxes only)
# Retained for reference; does NOT handle CVAT format.
# ---------------------------------------------------------------------------

class XMLToCOCOConverter:
    """Convert Pascal VOC XML annotations to COCO JSON format (bounding boxes only)."""

    def __init__(self, image_dir: str, annotation_dir: str):
        self.image_dir      = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.categories     = {}
        self.category_id_map = {}

    def register_categories(self, categories: Dict[str, int]):
        self.categories      = categories
        self.category_id_map = {v: k for k, v in categories.items()}

    def parse_xml(self, xml_path: str) -> Dict:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text
        size     = root.find("size")
        width    = int(size.find("width").text)
        height   = int(size.find("height").text)

        objects = []
        for obj in root.findall("object"):
            name   = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            objects.append(
                {"name": name, "bbox": [xmin, ymin, xmax, ymax],
                 "area": (xmax - xmin) * (ymax - ymin)}
            )

        return {"filename": filename, "width": width, "height": height, "objects": objects}

    def convert_to_coco(self, output_json: str, split: str = "train"):
        coco_dataset = {
            "info":        {"description": f"{split.upper()} Dataset", "version": "1.0", "year": 2024},
            "licenses":    [],
            "images":      [],
            "annotations": [],
            "categories":  [],
        }

        for cat_name, cat_id in self.categories.items():
            coco_dataset["categories"].append(
                {"id": cat_id, "name": cat_name, "supercategory": "defect"}
            )

        xml_files     = list(self.annotation_dir.glob("*.xml"))
        image_id      = 1
        annotation_id = 1

        for xml_file in tqdm(xml_files, desc=f"Converting {split}"):
            try:
                ann_data  = self.parse_xml(str(xml_file))
                img_files = [
                    f for f in self.image_dir.glob(f"{xml_file.stem}.*")
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff", ".tif")
                ]
                if not img_files:
                    continue

                img = cv2.imread(str(img_files[0]))
                if img is None:
                    continue

                height, width = img.shape[:2]
                coco_dataset["images"].append(
                    {"id": image_id,
                     "file_name": str(img_files[0].relative_to(self.image_dir.parent)),
                     "height": height, "width": width}
                )

                for obj in ann_data["objects"]:
                    if obj["name"] not in self.categories:
                        continue
                    xmin, ymin, xmax, ymax = obj["bbox"]
                    bw, bh = xmax - xmin, ymax - ymin
                    coco_dataset["annotations"].append(
                        {"id": annotation_id, "image_id": image_id,
                         "category_id": self.categories[obj["name"]],
                         "bbox": [xmin, ymin, bw, bh],
                         "area": bw * bh, "iscrowd": 0, "segmentation": []}
                    )
                    annotation_id += 1

                image_id += 1
            except Exception as e:
                logger.error("Error processing %s: %s", xml_file, e)

        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(coco_dataset, f, indent=2)

        logger.info("COCO dataset saved to %s (%d images, %d annotations)",
                    output_json, len(coco_dataset["images"]), len(coco_dataset["annotations"]))
        return coco_dataset


# ---------------------------------------------------------------------------
# Multi-modal image fusion
# ---------------------------------------------------------------------------

class MultimodalImageFusion:
    """Fuse RGB and Thermal images into 4-channel (RGBT) arrays."""

    @staticmethod
    def fuse_rgb_thermal(
        rgb_path: str,
        thermal_path: str,
        output_path: str,
        method: str = "concatenate",
    ):
        rgb     = cv2.imread(rgb_path)
        thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)

        if rgb is None or thermal is None:
            logger.error("Cannot read images: %s or %s", rgb_path, thermal_path)
            return None

        thermal = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]))

        if method == "concatenate":
            thermal_3ch = cv2.cvtColor(thermal, cv2.COLOR_GRAY2BGR)
            fused = np.concatenate([rgb, thermal_3ch], axis=2)
        elif method == "overlay":
            fused = rgb.copy()
            fused[:, :, 2] = cv2.addWeighted(fused[:, :, 2], 0.7, thermal, 0.3, 0)
        elif method == "weighted_average":
            thermal_3ch = cv2.cvtColor(thermal, cv2.COLOR_GRAY2BGR)
            fused = cv2.addWeighted(rgb, 0.5, thermal_3ch, 0.5, 0)
        else:
            raise ValueError(f"Unknown fusion method: {method}")

        cv2.imwrite(output_path, fused)
        return fused

    @staticmethod
    def create_4channel_dataset(rgb_dir: str, thermal_dir: str, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        rgb_files: List[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tiff", "*.tif"):
            rgb_files.extend(Path(rgb_dir).glob(ext))

        for rgb_file in tqdm(rgb_files, desc="Creating 4-channel images"):
            thermal_file = None
            for ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif"):
                candidate = Path(thermal_dir) / f"{rgb_file.stem}{ext}"
                if candidate.exists():
                    thermal_file = candidate
                    break

            if thermal_file is None:
                logger.warning("No thermal image found for %s", rgb_file.name)
                continue

            rgb     = cv2.imread(str(rgb_file))
            thermal = cv2.imread(str(thermal_file), cv2.IMREAD_GRAYSCALE)
            if rgb is None or thermal is None:
                continue

            thermal = cv2.resize(thermal, (rgb.shape[1], rgb.shape[0]))
            thermal_3ch = cv2.cvtColor(thermal, cv2.COLOR_GRAY2BGR)
            fused_4ch   = np.concatenate([rgb, thermal_3ch], axis=2)

            out_file = output_path / f"{rgb_file.stem}.npz"
            np.savez_compressed(str(out_file), image=fused_4ch)

        logger.info("4-channel dataset created in %s", output_dir)


# ---------------------------------------------------------------------------
# Dataset organizer
# ---------------------------------------------------------------------------

class DatasetOrganizer:
    """Organise a flat image directory into train / val / test sub-directories."""

    @staticmethod
    def organize_dataset(
        source_dir: str,
        output_dir: str,
        split_ratios: Optional[Dict[str, float]] = None,
    ):
        if split_ratios is None:
            split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

        source_path = Path(source_dir)
        output_path = Path(output_dir)

        for split in split_ratios:
            (output_path / split / "images").mkdir(parents=True, exist_ok=True)
            (output_path / split / "annotations").mkdir(parents=True, exist_ok=True)

        image_files: List[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tiff", "*.tif"):
            image_files.extend(source_path.glob(ext))
        np.random.shuffle(image_files)  # type: ignore[arg-type]

        total      = len(image_files)
        n_train    = int(total * split_ratios["train"])
        n_val      = int(total * (split_ratios["train"] + split_ratios["val"]))

        splits = {
            "train": image_files[:n_train],
            "val":   image_files[n_train:n_val],
            "test":  image_files[n_val:],
        }

        for split_name, files in splits.items():
            for img_file in tqdm(files, desc=f"Organising {split_name}"):
                dst = output_path / split_name / "images" / img_file.name
                os.link(str(img_file), str(dst))

        logger.info(
            "Dataset organised in %s | train=%d val=%d test=%d",
            output_dir,
            len(splits["train"]), len(splits["val"]), len(splits["test"]),
        )


# ---------------------------------------------------------------------------
# Quick-use example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Convert CVAT annotations to COCO JSON
    converter = CVATXMLToCOCOConverter("annotations.xml")
    converter.convert(
        output_train="dataset/instances_train.json",
        output_val="dataset/instances_val.json",
        train_ratio=0.8,
    )
