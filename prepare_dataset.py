#!/usr/bin/env python
"""
Data Preparation Script for Your Dataset
Converts your directory structure to COCO format
"""

import os
import shutil
import logging
from pathlib import Path
from dataset_utils import XMLToCOCOConverter, MultimodalImageFusion, DatasetOrganizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_your_dataset():
    """
    Prepare your specific dataset structure
    Adjust these paths based on your actual directory structure
    """
    
    # Your actual dataset paths
    RGB_DIR = "./RGB_FITTED"
    THERMAL_DIR = "./IRT_FITTED"
    FUSED_DIR = "./FUSED"
    ANNOTATIONS_FILE = "./annotations.xml"
    OUTPUT_DIR = "./dataset"
    
    logger.info("\n" + "=" * 70)
    logger.info("Data Preparation Pipeline")
    logger.info("=" * 70)
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # =========================================================================
    # STEP 1: Check what we have
    # =========================================================================
    logger.info("\nSTEP 1: Checking available data...")
    
    rgb_files = list(Path(RGB_DIR).glob('*.*')) if os.path.exists(RGB_DIR) else []
    thermal_files = list(Path(THERMAL_DIR).glob('*.*')) if os.path.exists(THERMAL_DIR) else []
    fused_files = list(Path(FUSED_DIR).glob('*.*')) if os.path.exists(FUSED_DIR) else []
    
    logger.info(f"  RGB images: {len(rgb_files)}")
    logger.info(f"  Thermal images: {len(thermal_files)}")
    logger.info(f"  Fused images: {len(fused_files)}")
    logger.info(f"  Annotation file exists: {os.path.exists(ANNOTATIONS_FILE)}")
    
    # =========================================================================
    # STEP 2: Choose which images to use
    # =========================================================================
    logger.info("\nSTEP 2: Choosing image source...")
    
    # Prefer RGBT fusion if both RGB and thermal available
    if rgb_files and thermal_files:
        logger.info("  ✓ Using RGB + Thermal fusion (4-channel RGBT)")
        use_type = "rgbt_fusion"
        image_source = RGB_DIR
    elif rgb_files:
        logger.info("  ✓ Using RGB images only")
        use_type = "rgb_only"
        image_source = RGB_DIR
    elif fused_files:
        logger.info("  ✓ Using pre-fused images")
        use_type = "fused_only"
        image_source = FUSED_DIR
    else:
        logger.error("  ✗ No images found! Check directory paths.")
        return False
    
    # =========================================================================
    # STEP 3: Copy images to dataset directory
    # =========================================================================
    logger.info("\nSTEP 3: Organizing images...")
    
    image_files = list(Path(image_source).glob('*.*'))
    image_files = [f for f in image_files if f.suffix.lower() in 
                   ['.jpg', '.jpeg', '.png', '.tiff', '.tif']]
    
    # Create train/val dirs
    train_dir = Path(OUTPUT_DIR) / 'train'
    val_dir = Path(OUTPUT_DIR) / 'val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Split images (70% train, 30% val)
    split_idx = int(len(image_files) * 0.7)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    logger.info(f"  Train split: {len(train_files)} images")
    logger.info(f"  Val split: {len(val_files)} images")
    
    # Copy files
    for img_file in train_files:
        try:
            dst = train_dir / img_file.name
            shutil.copy2(img_file, dst)
        except Exception as e:
            logger.warning(f"  Could not copy {img_file.name}: {e}")
    
    for img_file in val_files:
        try:
            dst = val_dir / img_file.name
            shutil.copy2(img_file, dst)
        except Exception as e:
            logger.warning(f"  Could not copy {img_file.name}: {e}")
    
    logger.info(f"  ✓ Images copied to {OUTPUT_DIR}")
    
    # =========================================================================
    # STEP 4: Create RGBT fusion if available
    # =========================================================================
    if use_type == "rgbt_fusion" and thermal_files:
        logger.info("\nSTEP 4: Creating 4-channel RGBT images...")
        try:
            rgbt_dir = Path(OUTPUT_DIR) / 'rgbt_4channel'
            MultimodalImageFusion.create_4channel_dataset(
                rgb_dir=str(image_source),
                thermal_dir=THERMAL_DIR,
                output_dir=str(rgbt_dir)
            )
            logger.info(f"  ✓ RGBT images saved to {rgbt_dir}")
        except Exception as e:
            logger.warning(f"  Could not create RGBT fusion: {e}")
    else:
        logger.info("\nSTEP 4: Skipping RGBT fusion (not available)")
    
    # =========================================================================
    # STEP 5: Convert XML annotations to COCO format
    # =========================================================================
    if os.path.exists(ANNOTATIONS_FILE):
        logger.info("\nSTEP 5: Converting annotations to COCO format...")
        try:
            # For now, assume all annotations are in one XML file
            # You may need to adjust based on your actual annotation structure
            
            # Create dummy converter to demonstrate
            # In reality, you may have multiple XML files or different format
            logger.warning("  ⚠ Note: Adjust annotation conversion for your exact format")
            logger.info("  Sample code to convert XML annotations:")
            logger.info("  ")
            logger.info("  from dataset_utils import XMLToCOCOConverter")
            logger.info("  converter = XMLToCOCOConverter(")
            logger.info(f"      image_dir='{OUTPUT_DIR}/train',")
            logger.info("      annotation_dir='path/to/xml/files'")
            logger.info("  )")
            logger.info("  converter.register_categories({")
            logger.info("      'cracks': 1,")
            logger.info("      'spalls': 2,")
            logger.info("      'moisture': 3")
            logger.info("  })")
            logger.info(f"  converter.convert_to_coco('{OUTPUT_DIR}/instances_train.json')")
            
        except Exception as e:
            logger.warning(f"  Could not process annotations: {e}")
    else:
        logger.info("\nSTEP 5: Annotations file not found")
        logger.info("  Please provide XML or JSON annotations in COCO format")
    
    # =========================================================================
    # STEP 6: Create minimal COCO JSON if annotations unavailable
    # =========================================================================
    logger.info("\nSTEP 6: Creating placeholder COCO JSON...")
    try:
        import json
        
        # Create minimal COCO files for training
        for split_name, split_dir in [('train', train_dir), ('val', val_dir)]:
            coco_data = {
                'info': {
                    'description': f'{split_name.upper()} Defect Detection Dataset',
                    'version': '1.0',
                    'year': 2024
                },
                'licenses': [],
                'images': [],
                'annotations': [],
                'categories': [
                    {'id': 0, 'name': 'background', 'supercategory': 'defect'},
                    {'id': 1, 'name': 'cracks', 'supercategory': 'defect'},
                    {'id': 2, 'name': 'spalls', 'supercategory': 'defect'},
                    {'id': 3, 'name': 'moisture', 'supercategory': 'defect'}
                ]
            }
            
            # Add image entries (placeholders without annotations)
            for idx, img_file in enumerate(split_dir.glob('*.*'), 1):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.tif']:
                    coco_data['images'].append({
                        'id': idx,
                        'file_name': f'{split_name}/{img_file.name}',
                        'height': 1024,  # Will be updated on load
                        'width': 1024    # Will be updated on load
                    })
            
            # Save placeholder COCO JSON
            output_json = Path(OUTPUT_DIR) / f'instances_{split_name}.json'
            with open(output_json, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            logger.info(f"  ✓ Placeholder COCO JSON created: instances_{split_name}.json")
    
    except Exception as e:
        logger.error(f"  ✗ Could not create COCO JSON: {e}")
        return False
    
    # =========================================================================
    # STEP 7: Update config.yaml
    # =========================================================================
    logger.info("\nSTEP 7: Configuration recommendations...")
    logger.info("  Update config.yaml with:")
    logger.info(f"    dataset:")
    logger.info(f"      root_dir: \"{OUTPUT_DIR}\"")
    logger.info(f"      train_annotation_file: \"instances_train.json\"")
    logger.info(f"      val_annotation_file: \"instances_val.json\"")
    logger.info(f"      num_classes: 4")
    
    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("✓ Data preparation complete!")
    logger.info("=" * 70)
    logger.info(f"\nDataset structure created at: {OUTPUT_DIR}")
    logger.info(f"  ├── train/")
    logger.info(f"  ├── val/")
    logger.info(f"  ├── instances_train.json")
    logger.info(f"  └── instances_val.json")
    
    if use_type == "rgbt_fusion":
        logger.info(f"  └── rgbt_4channel/  (4-channel RGBT images)")
    
    logger.info("\nNext steps:")
    logger.info("  1. Review COCO JSON files to ensure format is correct")
    logger.info("  2. Update config.yaml with dataset paths")
    logger.info("  3. Run: python CNNVIT.py")
    
    return True


if __name__ == "__main__":
    import sys
    
    logger.info("=" * 70)
    logger.info("Your Dataset Preparation Script")
    logger.info("=" * 70)
    
    success = prepare_your_dataset()
    
    if not success:
        logger.error("\n✗ Dataset preparation failed!")
        sys.exit(1)
    else:
        logger.info("\n✓ Ready for training!")
        sys.exit(0)
