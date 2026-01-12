"""
DeepFashion Dataset Downloader
==============================
Downloads and extracts the DeepFashion In-Shop Clothes Retrieval dataset.

The full dataset requires signing up at:
http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

For quick start, this script provides options to:
1. Download a pre-packaged subset from Kaggle
2. Use local images you already have
3. Download sample fashion images for testing
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import urllib.request
import zipfile
import gdown

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def download_kaggle_subset():
    """
    Download DeepFashion subset from Kaggle.
    Requires: pip install kaggle && kaggle.json in ~/.kaggle/
    
    Dataset: https://www.kaggle.com/datasets/nguyngiabol/deepfashion-inshop
    """
    print("\n" + "="*50)
    print(" Downloading DeepFashion from Kaggle")
    print("="*50)
    
    try:
        import kaggle
        
        # Create raw directory
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        print("\nDownloading dataset (this may take a while)...")
        kaggle.api.dataset_download_files(
            'nguyngiabol/deepfashion-inshop',
            path=str(RAW_DIR),
            unzip=True
        )
        print("✓ Download complete!")
        return True
        
    except ImportError:
        print("ERROR: kaggle package not installed.")
        print("Run: pip install kaggle")
        print("Then place your kaggle.json in ~/.kaggle/")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def download_sample_images():
    """
    Download a small sample of fashion images for testing.
    Uses freely available fashion images.
    """
    print("\n" + "="*50)
    print(" Downloading Sample Fashion Images")
    print("="*50)
    
    SAMPLE_DIR = RAW_DIR / "sample_fashion"
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Fashion MNIST or similar sample
    print("\nNote: For full training, you should download the complete DeepFashion dataset.")
    print("This sample is only for testing the pipeline.\n")
    
    # URLs for sample fashion images (using placeholder URLs)
    # In practice, you'd want to use actual fashion image sources
    
    print("Creating sample directory structure...")
    for category in ["tops", "bottoms", "dresses", "outerwear"]:
        (SAMPLE_DIR / category).mkdir(exist_ok=True)
    
    print("\n⚠ To add sample images:")
    print(f"  1. Download fashion images from the web")
    print(f"  2. Place them in: {SAMPLE_DIR}")
    print(f"  3. Organize by category (tops/, bottoms/, etc.)")
    print(f"\nOr download the full DeepFashion dataset using --source kaggle")
    
    return True


def download_from_gdrive(file_id: str, output_path: str):
    """Download a file from Google Drive."""
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)


def prepare_local_images(source_dir: str):
    """
    Prepare images from a local directory.
    Copies images to the raw data directory.
    """
    print("\n" + "="*50)
    print(" Preparing Local Images")
    print("="*50)
    
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        return False
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = RAW_DIR / "local_fashion"
    
    if dest_path.exists():
        shutil.rmtree(dest_path)
    
    # Copy images
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images_copied = 0
    
    dest_path.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(list(source_path.rglob('*')), desc="Copying images"):
        if img_path.suffix.lower() in image_extensions:
            # Flatten structure or maintain it
            dest_file = dest_path / img_path.name
            # Handle duplicates by adding number
            if dest_file.exists():
                stem = img_path.stem
                suffix = img_path.suffix
                counter = 1
                while dest_file.exists():
                    dest_file = dest_path / f"{stem}_{counter}{suffix}"
                    counter += 1
            shutil.copy2(img_path, dest_file)
            images_copied += 1
    
    print(f"\n✓ Copied {images_copied} images to {dest_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare DeepFashion dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_deepfashion.py --source kaggle
  python download_deepfashion.py --source local --local-path /path/to/images
  python download_deepfashion.py --source sample
        """
    )
    
    parser.add_argument(
        '--source', 
        type=str, 
        choices=['kaggle', 'local', 'sample'],
        default='sample',
        help='Data source: kaggle (full dataset), local (your images), sample (test images)'
    )
    
    parser.add_argument(
        '--local-path',
        type=str,
        help='Path to local images (required if --source local)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print(" DeepFashion Dataset Downloader")
    print(" For E-Commerce Image Generation")
    print("="*50)
    
    # Create directories
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    
    success = False
    
    if args.source == 'kaggle':
        success = download_kaggle_subset()
    elif args.source == 'local':
        if not args.local_path:
            print("ERROR: --local-path required when using --source local")
            sys.exit(1)
        success = prepare_local_images(args.local_path)
    elif args.source == 'sample':
        success = download_sample_images()
    
    if success:
        print("\n" + "="*50)
        print(" Next Steps")
        print("="*50)
        print("\n1. Run the data preparation script:")
        print("   python data/prepare_dataset.py")
        print("\n2. Start training:")
        print("   python projected_gan/train.py")
        print("="*50 + "\n")
    else:
        print("\n❌ Download failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
