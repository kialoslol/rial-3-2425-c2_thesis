"""
Quick Start Guide - Dataset Conversion & Training Pipeline
Run this script to convert your XML dataset to COCO format and start training
"""

import os
import sys
from pathlib import Path
import yaml
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories(root_dir: str = "."):
    """Create necessary directories"""
    dirs = [
        'checkpoints',
        'visualizations',
        'predictions',
        'logs',
        'dataset/train',
        'dataset/val',
        'dataset/test'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {d}")


def convert_xml_dataset(
    image_dir: str,
    xml_dir: str,
    output_dir: str = "dataset",
    defect_categories: dict = None
):
    """
    Convert XML annotations to COCO format
    
    Args:
        image_dir: Directory containing images
        xml_dir: Directory containing XML annotation files
        output_dir: Output directory for COCO JSON files
        defect_categories: Map of defect names to IDs
    """
    
    if defect_categories is None:
        defect_categories = {
            'cracks': 1,
            'spalls': 2,
            'moisture': 3
        }
    
    from dataset_utils import XMLToCOCOConverter
    
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Converting XML to COCO Format")
    logger.info("="*60)
    
    converter = XMLToCOCOConverter(
        image_dir=image_dir,
        annotation_dir=xml_dir
    )
    
    converter.register_categories(defect_categories)
    
    # Convert training set
    train_json = os.path.join(output_dir, 'instances_train.json')
    logger.info(f"\nConverting training annotations...")
    logger.info(f"  Image directory: {image_dir}")
    logger.info(f"  Annotation directory: {xml_dir}")
    logger.info(f"  Output: {train_json}")
    
    converter.convert_to_coco(output_json=train_json, split='train')
    
    logger.info("\n✓ XML to COCO conversion complete!")
    return train_json


def fuse_rgb_thermal(
    rgb_dir: str,
    thermal_dir: str,
    output_dir: str = "dataset/4channel"
):
    """
    Optionally fuse RGB and thermal images into 4-channel RGBT
    
    Args:
        rgb_dir: Directory with RGB images
        thermal_dir: Directory with thermal images
        output_dir: Output directory for 4-channel images
    """
    
    from dataset_utils import MultimodalImageFusion
    
    logger.info("\n" + "="*60)
    logger.info("STEP 2 (Optional): Fusing RGB + Thermal Images")
    logger.info("="*60)
    
    logger.info(f"\nFusing Images...")
    logger.info(f"  RGB directory: {rgb_dir}")
    logger.info(f"  Thermal directory: {thermal_dir}")
    logger.info(f"  Output directory: {output_dir}")
    
    try:
        MultimodalImageFusion.create_4channel_dataset(
            rgb_dir=rgb_dir,
            thermal_dir=thermal_dir,
            output_dir=output_dir
        )
        logger.info("\n✓ Image fusion complete!")
        return output_dir
    except Exception as e:
        logger.warning(f"\n⚠ Skipping image fusion: {e}")
        return None


def train_model(config_path: str = "config.yaml"):
    """
    Start training the MobileViT model
    
    Args:
        config_path: Path to configuration file
    """
    
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Training MobileViT Model")
    logger.info("="*60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    from CNNVIT import DefectDetectionPipeline
    
    logger.info("\nTraining Configuration:")
    logger.info(f"  Dataset root: {config['dataset']['root_dir']}")
    logger.info(f"  Number of classes: {config['dataset']['num_classes']}")
    logger.info(f"  Batch size: {config['training']['batch_size']}")
    logger.info(f"  Number of epochs: {config['training']['num_epochs']}")
    logger.info(f"  Image size: {config['dataset']['image_size']}")
    logger.info(f"  Learning rate: {config['training']['learning_rate']}")
    
    # Initialize pipeline
    pipeline = DefectDetectionPipeline(
        dataset_root=config['dataset']['root_dir'],
        num_classes=config['dataset']['num_classes'],
        batch_size=config['training']['batch_size'],
        num_epochs=config['training']['num_epochs'],
        image_size=config['dataset']['image_size'],
        checkpoint_dir=config['checkpoint']['save_dir']
    )
    
    # Train
    logger.info("\nStarting training...\n")
    history = pipeline.train(
        train_annotation_file=config['dataset']['train_annotation_file'],
        val_annotation_file=config['dataset'].get('val_annotation_file', 'instances_val.json')
    )
    
    logger.info("\n✓ Training complete!")
    return pipeline, history


def visualize_results(history, output_dir: str = "visualizations"):
    """
    Generate training visualizations
    
    Args:
        history: Training history dictionary
        output_dir: Output directory for visualizations
    """
    
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Generating Visualizations")
    logger.info("="*60)
    
    from visualization import DefectVisualization
    
    viz = DefectVisualization(output_dir=output_dir)
    
    logger.info(f"\nGenerating plots...")
    logger.info(f"  Output directory: {output_dir}")
    
    # Plot training history
    viz.plot_training_history(history, save_path=os.path.join(output_dir, 'training_history.png'))
    
    logger.info("\n✓ Visualizations generated!")
    return viz


def run_inference(
    pipeline,
    image_path: str,
    checkpoint_path: str = None,
    output_dir: str = "predictions"
):
    """
    Run inference on a test image
    
    Args:
        pipeline: Trained pipeline
        image_path: Path to test image
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for predictions
    """
    
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Running Inference")
    logger.info("="*60)
    
    logger.info(f"\nRunning inference on: {image_path}")
    
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return None
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    prediction = pipeline.predict(
        image_path=image_path,
        checkpoint_path=checkpoint_path
    )
    
    logger.info(f"✓ Prediction complete!")
    logger.info(f"  Mask shape: {prediction['mask'].shape}")
    logger.info(f"  Image shape: {prediction['image'].shape}")
    
    return prediction


def main():
    """Main execution flow"""
    
    logger.info("\n")
    logger.info("╔" + "="*58 + "╗")
    logger.info("║" + " "*58 + "║")
    logger.info("║" + "  MobileViT V3 Defect Detection - Quick Start".center(58) + "║")
    logger.info("║" + " "*58 + "║")
    logger.info("╚" + "="*58 + "╝")
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    
    # Update these paths to your dataset locations
    IMAGE_DIR = r"C:\Users\David\Documents\DLSU\THESIS\images"
    XML_DIR = r"C:\Users\David\Documents\DLSU\THESIS\annotations"
    RGB_DIR = r"C:\Users\David\Documents\DLSU\THESIS\rgb"  # Optional
    THERMAL_DIR = r"C:\Users\David\Documents\DLSU\THESIS\thermal"  # Optional
    OUTPUT_DIR = "dataset"
    
    # =========================================================================
    # EXECUTION
    # =========================================================================
    
    try:
        # Step 0: Setup directories
        setup_directories()
        
        # Step 1: Convert XML to COCO
        if os.path.exists(IMAGE_DIR) and os.path.exists(XML_DIR):
            convert_xml_dataset(
                image_dir=IMAGE_DIR,
                xml_dir=XML_DIR,
                output_dir=OUTPUT_DIR
            )
        else:
            logger.warning("⚠ Image or XML directory not found. Skipping conversion.")
        
        # Step 2: (Optional) Fuse RGB + Thermal
        # Uncomment if you have thermal images
        # if os.path.exists(RGB_DIR) and os.path.exists(THERMAL_DIR):
        #     fuse_rgb_thermal(RGB_DIR, THERMAL_DIR)
        
        # Step 3: Train model
        pipeline, history = train_model(config_path="config.yaml")
        
        # Step 4: Visualize results
        visualize_results(history)
        
        # Step 5: Run inference (optional)
        # Uncomment and update path to test image
        # test_image_path = "path/to/test/image.jpg"
        # prediction = run_inference(
        #     pipeline,
        #     image_path=test_image_path,
        #     checkpoint_path="./checkpoints/best_model_epoch_0.pth"
        # )
        
        logger.info("\n" + "="*60)
        logger.info("✓ PIPELINE COMPLETE!")
        logger.info("="*60)
        logger.info("\nYour trained model has been saved to: ./checkpoints/")
        logger.info("Training visualizations saved to: ./visualizations/")
        logger.info("Predictions will be saved to: ./predictions/")
        
    except Exception as e:
        logger.error(f"\n✗ Error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
