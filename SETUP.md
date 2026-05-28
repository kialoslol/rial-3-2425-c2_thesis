# Project Setup & Installation Guide

## Project Structure

```
THESIS/
├── CNNVIT.py                   # Main model, trainer, and pipeline
├── dataset.py                  # Dataset loaders (COCOSegmentationDataset)
├── dataset_utils.py            # Dataset converters & formatters
│                               # - XMLToCOCOConverter
│                               # - MultimodalImageFusion
│                               # - DatasetOrganizer
├── visualization.py            # Visualization & plotting utilities
├── config.yaml                 # Training configuration
├── requirements.txt            # Python dependencies
├── quickstart.py               # Automated pipeline script
├── __init__.py                 # Package initialization
│
├── README.md                   # Full documentation
├── DATASET_FORMAT_GUIDE.md    # Dataset format specifications
├── SETUP.md                    # This file
│
├── checkpoints/                # Saved model checkpoints
├── visualizations/             # Output visualizations
├── predictions/                # Inference outputs
└── dataset/                    # COCO formatted dataset
    ├── train/
    ├── val/
    ├── instances_train.json
    └── instances_val.json
```

## File Breakdown

### Core Model Files

#### `CNNVIT.py` - Main Model & Training
- **Classes**:
  - `MobileViTSegmentationHead`: Decoder head for segmentation
  - `MobileViTSegmentationModel`: Full MobileViT model
  - `DefectDetectionMetrics`: Metric computation (Accuracy, IoU, Dice, F1, mAP)
  - `DefectDetectionTrainer`: Training & validation loop
  - `DefectDetectionPipeline`: Complete training + inference pipeline

#### `dataset.py` - Dataset Loading (NEW)
- **Classes**:
  - `RandomGaussianNoise`: Data augmentation
  - `COCOSegmentationDataset`: COCO dataset loader with comprehensive augmentation
  
- **Features**:
  - Loads COCO JSON annotations
  - Generates segmentation masks
  - Multi-level augmentation (geometric, color, noise)
  - Handles JPEG, PNG, TIFF formats

#### `dataset_utils.py` - Dataset Formatting
- **Classes**:
  - `XMLToCOCOConverter`: Converts Pascal VOC XML to COCO JSON format
  - `MultimodalImageFusion`: Fuses RGB + Thermal images into 4-channel RGBT
  - `DatasetOrganizer`: Organizes dataset into train/val/test splits

#### `visualization.py` - Results Visualization
- **Classes**:
  - `DefectVisualization`: Plotting & visualization tools
    - Training history plots
    - Confusion matrix
    - Segmentation results
    - Defect distribution

### Configuration & Utilities

- **config.yaml**: All training hyperparameters
- **requirements.txt**: Python dependencies
- **quickstart.py**: Automated end-to-end pipeline

## Installation

### 1. Clone/Download Project
```bash
cd c:\Users\David\Documents\DLSU\THESIS
```

### 2. Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/Scripts/activate  # Windows
# or: source venv/bin/activate  # Linux/Mac

# OR using conda
conda create -n defect_detection python=3.10
conda activate defect_detection
```

### 3. Install Dependencies
```bash
# Install all requirements
pip install -r requirements.txt

# OR install individually
pip install torch torchvision timm pycocotools opencv-python pyyaml matplotlib
```

### 4. Verify Installation
```bash
python -c "import torch; import timm; print('✓ Installation successful')"
```

## Quick Start

### Option 1: Using Automated Pipeline
```bash
python quickstart.py
```

### Option 2: Manual Steps

#### Step 1: Prepare Dataset
Convert your XML annotations to COCO format:
```python
from dataset_utils import XMLToCOCOConverter

converter = XMLToCOCOConverter(
    image_dir="path/to/images",
    annotation_dir="path/to/xml"
)
converter.register_categories({
    'cracks': 1,
    'spalls': 2,
    'moisture': 3
})
converter.convert_to_coco("instances_train.json")
```

#### Step 2: Optionally Fuse RGB + Thermal
```python
from dataset_utils import MultimodalImageFusion

MultimodalImageFusion.create_4channel_dataset(
    rgb_dir="path/to/rgb",
    thermal_dir="path/to/thermal",
    output_dir="dataset/4channel"
)
```

#### Step 3: Update config.yaml
```yaml
dataset:
  root_dir: "path/to/your/dataset"
  train_annotation_file: "instances_train.json"
  val_annotation_file: "instances_val.json"
  num_classes: 4
```

#### Step 4: Train Model
```python
from CNNVIT import DefectDetectionPipeline

pipeline = DefectDetectionPipeline(
    dataset_root="dataset",
    num_classes=4,
    batch_size=8,
    num_epochs=100
)

history = pipeline.train()
```

#### Step 5: Visualize Results
```python
from visualization import DefectVisualization

viz = DefectVisualization()
viz.plot_training_history(history)
```

#### Step 6: Run Inference
```python
prediction = pipeline.predict(
    "path/to/test/image.jpg",
    checkpoint_path="checkpoints/best_model_epoch_0.pth"
)
```

## Usage Examples

### Example 1: Complete Training Pipeline
```python
import yaml
from CNNVIT import DefectDetectionPipeline
from visualization import DefectVisualization

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Train
pipeline = DefectDetectionPipeline(
    dataset_root=config['dataset']['root_dir'],
    num_classes=config['dataset']['num_classes'],
    batch_size=config['training']['batch_size'],
    num_epochs=config['training']['num_epochs']
)

history = pipeline.train(
    train_annotation_file=config['dataset']['train_annotation_file'],
    val_annotation_file=config['dataset']['val_annotation_file']
)

# Visualize
viz = DefectVisualization()
viz.plot_training_history(history)
```

### Example 2: Single Image Inference
```python
from CNNVIT import DefectDetectionPipeline

pipeline = DefectDetectionPipeline(
    dataset_root="dataset",
    num_classes=4
)

result = pipeline.predict(
    image_path="test_image.jpg",
    checkpoint_path="checkpoints/best_model_epoch_0.pth"
)

print(f"Predicted mask shape: {result['mask'].shape}")
```

### Example 3: Batch Inference
```python
results = pipeline.batch_predict(
    image_dir="test_images",
    checkpoint_path="checkpoints/best_model_epoch_0.pth",
    output_dir="predictions"
)
```

## GPU Acceleration

### Check GPU Availability
```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

### Enable GPU
The model automatically uses GPU if available. To force CPU:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU
```

## Troubleshooting

### Issue: Module not found error
```
ModuleNotFoundError: No module named 'dataset'
```
**Fix**: Ensure `dataset.py` is in the same directory as `CNNVIT.py`

### Issue: COCO annotation not found
```
FileNotFoundError: instances_train.json not found
```
**Fix**: Generate COCO JSON using `XMLToCOCOConverter` first

### Issue: Out of memory (OOM)
**Fix**: Reduce `batch_size` in config.yaml or reduce `image_size`

### Issue: Dataset loading is slow
**Fix**: 
- Increase `num_workers` in DataLoader
- Use SSD storage if available
- Pre-process images to smaller sizes

## Performance Tips

1. **Use TIFF images** for best thermal + RGB fusion results
2. **Enable mixed precision training** (set in config.yaml)
3. **Use multiple GPUs** with DataParallel
4. **Increase augmentation** for better generalization
5. **Use learning rate scheduling** (already enabled)

## Model Metrics

The model computes and tracks:
- ✅ **Accuracy**: Pixel-level classification accuracy
- ✅ **IoU**: Intersection over Union
- ✅ **Dice**: Dice coefficient
- ✅ **F1-Score**: Harmonic mean of precision/recall
- ✅ **mAP**: Mean Average Precision

## Citation

If you use this model in your thesis:
```bibtex
@article{mehta2021mobilevit,
  title={MobileViT: Light-weight Vision Transformers},
  author={Mehta, Sachin and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2110.02178},
  year={2021}
}
```

## Support & Debugging

1. Check `config.yaml` for correct paths
2. Verify COCO JSON format matches specification
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Enable debug logging: Set `logging_level: DEBUG` in config
5. Check `logs/training.log` for detailed error messages

## Next Steps

1. ✅ Install dependencies
2. ✅ Prepare dataset (convert XML to COCO)
3. ✅ Run training (adjust config.yaml as needed)
4. ✅ Visualize results
5. ✅ Deploy inference on new images

For detailed dataset format information, see `DATASET_FORMAT_GUIDE.md`
