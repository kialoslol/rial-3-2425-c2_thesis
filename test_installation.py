#!/usr/bin/env python
"""
Test Script - Verify installation and basic functionality
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test if all required packages are installed"""
    logger.info("\n" + "=" * 70)
    logger.info("Testing Imports...")
    logger.info("=" * 70)
    
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("timm", "timm"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("pycocotools", "pycocotools"),
        ("yaml", "PyYAML"),
        ("matplotlib", "Matplotlib"),
    ]
    
    failed = []
    
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            logger.info(f"  ✓ {display_name}")
        except ImportError as e:
            logger.error(f"  ✗ {display_name}: {e}")
            failed.append(display_name)
    
    if failed:
        logger.error(f"\nMissing packages: {', '.join(failed)}")
        logger.error(f"Install with: pip install {' '.join(failed).lower()}")
        return False
    
    logger.info("\n✓ All packages installed!")
    return True


def test_local_modules():
    """Test if local modules can be imported"""
    logger.info("\n" + "=" * 70)
    logger.info("Testing Local Modules...")
    logger.info("=" * 70)
    
    modules = [
        ("dataset", "Dataset module"),
        ("CNNVIT", "Model module"),
        ("dataset_utils", "Dataset utilities"),
        ("visualization", "Visualization module"),
    ]
    
    failed = []
    
    for module_name, display_name in modules:
        try:
            __import__(module_name)
            logger.info(f"  ✓ {display_name}")
        except ImportError as e:
            logger.error(f"  ✗ {display_name}: {e}")
            failed.append(display_name)
    
    if failed:
        logger.error(f"\nMissing modules: {', '.join(failed)}")
        logger.error(f"Ensure files are in current directory: {', '.join([m[0] + '.py' for m in modules])}")
        return False
    
    logger.info("\n✓ All local modules found!")
    return True


def test_gpu():
    """Test GPU availability"""
    logger.info("\n" + "=" * 70)
    logger.info("Testing GPU...")
    logger.info("=" * 70)
    
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"  ✓ GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"    CUDA Version: {torch.version.cuda}")
            logger.info(f"    cuDNN Version: {torch.backends.cudnn.version()}")
        else:
            logger.warning("  ⚠ GPU not available (CPU will be used)")
        return True
    except Exception as e:
        logger.error(f"  ✗ GPU test failed: {e}")
        return False


def test_config():
    """Test config file"""
    logger.info("\n" + "=" * 70)
    logger.info("Testing Configuration...")
    logger.info("=" * 70)
    
    try:
        import yaml
        from pathlib import Path
        
        config_path = Path("config.yaml")
        if not config_path.exists():
            logger.error(f"  ✗ config.yaml not found")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['dataset', 'model', 'training', 'checkpoint']
        missing = [s for s in required_sections if s not in config]
        
        if missing:
            logger.error(f"  ✗ Missing config sections: {missing}")
            return False
        
        logger.info(f"  ✓ config.yaml found and valid")
        logger.info(f"    - Dataset root: {config['dataset']['root_dir']}")
        logger.info(f"    - Batch size: {config['training']['batch_size']}")
        logger.info(f"    - Epochs: {config['training']['num_epochs']}")
        
        return True
    
    except Exception as e:
        logger.error(f"  ✗ Config test failed: {e}")
        return False


def test_dataset_structure():
    """Test dataset directory structure"""
    logger.info("\n" + "=" * 70)
    logger.info("Testing Dataset Structure...")
    logger.info("=" * 70)
    
    try:
        from pathlib import Path
        
        # Check reference dataset directories
        dataset_dirs = [
            "RGB_FITTED",
            "IRT_FITTED",
            "FUSED"
        ]
        
        found = []
        missing = []
        
        for dir_name in dataset_dirs:
            dir_path = Path(dir_name)
            if dir_path.exists() and (list(dir_path.glob('*.*'))):
                count = len(list(dir_path.glob('*.*')))
                logger.info(f"  ✓ {dir_name}: {count} files")
                found.append(dir_name)
            else:
                missing.append(dir_name)
        
        if found:
            logger.info(f"\n  Found source data: {', '.join(found)}")
            logger.info(f"  Next: Run 'python prepare_dataset.py' to prepare dataset")
        else:
            logger.warning(f"\n  ⚠ No source data directories found")
            logger.info(f"  Place your images in: {', '.join(dataset_dirs)}")
        
        return True
    
    except Exception as e:
        logger.error(f"  ✗ Dataset test failed: {e}")
        return False


def test_model_creation():
    """Test model instantiation"""
    logger.info("\n" + "=" * 70)
    logger.info("Testing Model Creation...")
    logger.info("=" * 70)
    
    try:
        import torch
        from CNNVIT import MobileViTSegmentationModel
        
        logger.info("  Creating model...")
        model = MobileViTSegmentationModel(num_classes=4, pretrained=False)
        
        logger.info(f"  ✓ Model created successfully")
        logger.info(f"    Model type: {type(model).__name__}")
        
        # Test forward pass
        logger.info("  Testing forward pass...")
        dummy_input = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            output = model(dummy_input)
        
        logger.info(f"  ✓ Forward pass successful")
        logger.info(f"    Input shape: {dummy_input.shape}")
        logger.info(f"    Output shape: {output.shape}")
        
        return True
    
    except Exception as e:
        logger.error(f"  ✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 70)
    logger.info("Installation & Setup Verification")
    logger.info("=" * 70)
    
    tests = [
        ("Dependencies", test_imports),
        ("Local Modules", test_local_modules),
        ("GPU", test_gpu),
        ("Configuration", test_config),
        ("Dataset Structure", test_dataset_structure),
        ("Model Creation", test_model_creation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"\n✗ {test_name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Test Summary")
    logger.info("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n" + "=" * 70)
        logger.info("✓ All tests passed! Ready to train!")
        logger.info("=" * 70)
        logger.info("\nNext steps:")
        logger.info("  1. Prepare your dataset: python prepare_dataset.py")
        logger.info("  2. Update config.yaml with your paths")
        logger.info("  3. Start training: python CNNVIT.py")
        logger.info("\nOr run automated pipeline: python quickstart.py")
        return 0
    else:
        logger.error("\n" + "=" * 70)
        logger.error("✗ Some tests failed. Fix issues above and try again.")
        logger.error("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
