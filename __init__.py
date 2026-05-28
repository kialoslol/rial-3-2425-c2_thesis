"""
MobileViT V3 Hybrid CNN-ViT Model for Defect Detection
Segmentation-based defect detection supporting COCO format datasets
"""

__version__ = "1.0.0"
__author__ = "Thesis Development"

from dataset import COCOSegmentationDataset, RandomGaussianNoise

__all__ = [
    'COCOSegmentationDataset',
    'RandomGaussianNoise',
]
