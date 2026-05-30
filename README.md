# MobileViT V3 Hybrid CNN-ViT Defect Detection Model

## Overview
This project implements a **MobileViT V3-based semantic segmentation model** for real-world defect detection (cracks, spalls, moisture) with comprehensive metrics including accuracy, mAP, IoU, F1-score, and Dice coefficient.

### Architecture
- **Backbone**: MobileViTv2_100 (fusion of CNN + Vision Transformer)
- **Task**: Semantic segmentation + Instance detection
- **Input**: COCO-formatted dataset (supports RGB + Thermal fusion)
- **Output**: Per-pixel segmentation masks with multi-modal analysis

---

## Dataset Setup Guide

### **Your Current Dataset Structure**
You have:
- `RGB` images
- `Thermal` images  
- `Fused` images
- `XML` files (annotations)
- `ZIP` files

### **Required COCO Format**
The model requires **COCO JSON annotation format**:

```
dataset/
├── train/
│   ├── *.jpg, *.png, *.tiff, *.tif  (images)
│   └── (optional) *.xml (will be converted)
├── val/
│   ├── *.jpg, *.png, *.tiff, *.tif
│   └── (optional) *.xml
├── instances_train.json    ← COCO annotations
└── instances_val.json      ← COCO annotations
```

### **COCO JSON Format Example**
```json
{
  "info": {
    "description": "Defect Detection Dataset",
    "version": "1.0",
    "year": 2024
  },
  "licenses": [],
  "images": [
    {
      "id": 1,
      "file_name": "train/image001.jpg",
      "height": 1024,
      "width": 1024
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 250],
      "area": 50000,
      "iscrowd": 0,
      "segmentation": []
    }
  ],
  "categories": [
    {"id": 0, "name": "background", "supercategory": "defect"},
    {"id": 1, "name": "cracks", "supercategory": "defect"},
    {"id": 2, "name": "spalls", "supercategory": "defect"},
    {"id": 3, "name": "moisture", "supercategory": "defect"}
  ]
}
```

### **Step 1: Convert XML Annotations to COCO**

If your annotations are in **Pascal VOC XML format**:

```python
from dataset_utils import XMLToCOCOConverter

# Initialize converter
converter = XMLToCOCOConverter(
    image_dir="path/to/your/images",
    annotation_dir="path/to/your/xml_annotations"
)

# Register defect categories
converter.register_categories({
    'cracks': 1,
    'spalls': 2,
    'moisture': 3
})

# Convert to COCO format
converter.convert_to_coco(
    output_json="instances_train.json",
    split='train'
)

converter.convert_to_coco(
    output_json="instances_val.json",
    split='val'
)
```

### **Step 2: Fuse RGB + Thermal Images (Optional)**

If using both RGB and thermal images:

```python
from dataset_utils import MultimodalImageFusion

# Create 4-channel RGBT images
MultimodalImageFusion.create_4channel_dataset(
    rgb_dir="path/to/rgb_images",
    thermal_dir="path/to/thermal_images",
    output_dir="path/to/output_4channel"
)
```

This creates **4-channel images** (B, G, R, Thermal) for better defect detection.

### **Step 3: Organize Dataset**

```python
from dataset_utils import DatasetOrganizer

DatasetOrganizer.organize_dataset(
    source_dir="path/to/all/images",
    output_dir="path/to/organized/dataset",
    split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}
)
```

---

## Installation

### **Requirements**
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)

### **Step 1: Install Dependencies**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install timm pycocotools opencv-python numpy tqdm pyyaml matplotlib seaborn scikit-learn
```

### **Step 2: Verify Installation**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Configuration

Edit `config.yaml` for your dataset and training parameters:

```yaml
dataset:
  root_dir: "c:/Users/David/Documents/DLSU/THESIS/dataset"
  train_annotation_file: "instances_train.json"
  val_annotation_file: "instances_val.json"
  image_size: 512
  num_classes: 4

training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001

augmentation:
  enable: true
  random_flip: true
  random_rotation: true
  color_jitter: true
  gaussian_blur: true
  gaussian_noise: true
```

---

## Data Augmentation

The model uses **comprehensive augmentation**:
- ✅ **Geometric**: Random flips (H/V), rotations (±15°), affine transforms
- ✅ **Color**: Brightness, contrast, saturation, hue jittering
- ✅ **Blur**: Gaussian blur with random kernel sizes
- ✅ **Noise**: Gaussian noise injection for robustness

These improve model generalization and reduce overfitting on small datasets.

---

## Training

### **Basic Training**

```python
from CNNVIT import DefectDetectionPipeline

pipeline = DefectDetectionPipeline(
    dataset_root="c:/path/to/dataset",
    num_classes=4,
    batch_size=8,
    num_epochs=100,
    image_size=512
)

history = pipeline.train(
    train_annotation_file='instances_train.json',
    val_annotation_file='instances_val.json'
)
```

### **With Configuration File**

```python
import yaml
from CNNVIT import DefectDetectionPipeline

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

pipeline = DefectDetectionPipeline(
    dataset_root=config['dataset']['root_dir'],
    num_classes=config['dataset']['num_classes'],
    batch_size=config['training']['batch_size'],
    num_epochs=config['training']['num_epochs']
)

history = pipeline.train()
```

---

## Inference

### **Single Image Prediction**

```python
prediction = pipeline.predict(
    image_path='path/to/test/image.jpg',
    checkpoint_path='./checkpoints/best_model_epoch_0.pth'
)

# prediction contains:
# - mask: segmentation mask with class IDs
# - image: original image
# - image_shape: original dimensions
```

### **Batch Prediction**

```python
results = pipeline.batch_predict(
    image_dir='path/to/test/images',
    checkpoint_path='./checkpoints/best_model_epoch_0.pth',
    output_dir='./predictions'
)
```

---

## Metrics & Visualization

### **Available Metrics**
- **Accuracy**: Pixel-level classification accuracy
- **IoU (mIoU)**: Intersection over Union for segmentation
- **Dice Coefficient**: F1-like metric for defect regions
- **F1-Score**: Harmonic mean of precision and recall
- **mAP**: Mean Average Precision (COCO style)

### **Visualizations**

```python
from visualization import DefectVisualization

viz = DefectVisualization(output_dir="./visualizations")

# Plot training history
viz.plot_training_history(history)

# Plot confusion matrix
viz.plot_confusion_matrix(trainer.metrics.confusion_matrix)

# Plot segmentation results
viz.plot_segmentation_results(
    image=image,
    ground_truth_mask=gt_mask,
    predicted_mask=pred_mask
)

# Compare metrics
metrics = {
    'accuracy': 0.92,
    'iou': 0.87,
    'dice': 0.89,
    'f1_score': 0.88,
    'map': 0.85
}
viz.plot_metrics_comparison(metrics)

# Defect distribution
viz.plot_defect_distribution(all_masks)
```

---

## Project Structure

```
THESIS/
├── CNNVIT.py                 # Main model architecture
├── dataset_utils.py          # Dataset conversion utilities
├── visualization.py          # Visualization tools
├── config.yaml              # Configuration file
├── README.md                # This file
├── checkpoints/             # Saved models
├── visualizations/          # Output visualizations
├── predictions/             # Inference outputs
└── dataset/                 # Your COCO dataset
    ├── train/
    ├── val/
    ├── instances_train.json
    └── instances_val.json
```

---

## Workflow Summary

1. **Prepare Dataset**
   ```bash
   # Convert XML to COCO JSON
   python -c "from dataset_utils import XMLToCOCOConverter; ..."
   ```

2. **Update config.yaml** with your dataset path and parameters

3. **Train Model**
   ```bash
   python CNNVIT.py
   ```

4. **Visualize Results**
   ```bash
   python -c "from visualization import DefectVisualization; ..."
   ```

5. **Run Inference**
   ```python
   pipeline.predict('test_image.jpg', 'checkpoints/best_model.pth')
   ```

---

## GPU Acceleration

Enable CUDA for faster training:

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Automatically uses GPU if available
pipeline = DefectDetectionPipeline(...)
```

---

## Troubleshooting

### **Dataset Not Found**
- Ensure `instances_train.json` and `instances_val.json` exist in dataset root
- Verify file paths use forward slashes `/` or raw strings `r"path"`

### **OOM Error (Out of Memory)**
- Reduce `batch_size` in config.yaml
- Reduce `image_size` (e.g., 256 or 384)

### **Low Metrics**
- Check data augmentation is properly configured
- Verify annotation format matches COCO standard
- Increase number of epochs or adjust learning rate

### **Slow Training**
- Use GPU (CUDA): Check `torch.cuda.is_available()`
- Reduce `num_workers` if causing issues
- Use mixed precision: Set `mixed_precision: true` in config

---

## Advanced Features

### **Multi-GPU Training**
```python
model = nn.DataParallel(model)
```

### **Mixed Precision Training**
Enable in config.yaml for faster training with lower memory:
```yaml
device:
  mixed_precision: true
```

### **Custom Loss Functions**
Modify `MobileViTSegmentationModel` to use:
- Focal Loss for imbalanced classes
- Dice Loss for better segmentation
- Lovasz Loss for IoU optimization

---

## References

- [MobileViT Paper](https://arxiv.org/abs/2110.02178)
- [COCO Dataset Format](https://cocodataset.org/#format-data)
- [PyTorch Segmentation](https://pytorch.org/vision/stable/segmentation.html)
- [timm Models](https://github.com/rwightman/pytorch-image-models)

---

## License & Citation

If you use this model in your thesis, please cite:

```bibtex
@article{mehta2021mobilevit,
  title={MobileViT: Light-weight Vision Transformers},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2110.02178},
  year={2021}
}
```

---

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify dataset format matches examples
3. Enable debug logging: Set `logging.level: DEBUG` in config.yaml

