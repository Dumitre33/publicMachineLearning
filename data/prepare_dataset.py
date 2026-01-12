"""
DeepFashion Dataset Preparation
================================
Prepares the DeepFashion dataset for:
1. Projected GAN training (unconditional generation)
2. Stable Diffusion LoRA training (text-conditioned generation)

Handles:
- Image resizing and normalization
- Dataset splitting (train/val)
- Caption generation for LoRA
- Creating proper folder structure
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Output directories
GAN_DATA_DIR = PROCESSED_DIR / "projected_gan"
LORA_DATA_DIR = PROCESSED_DIR / "lora"


def get_fashion_caption(image_path: Path, category: str = None) -> str:
    """
    Generate a caption for a fashion image.
    Used for LoRA training with text conditioning.
    
    In practice, you might want to:
    - Use BLIP/CLIP to generate captions
    - Use existing annotations from DeepFashion
    - Manually label images
    """
    # Simple rule-based captioning
    filename = image_path.stem.lower()
    
    # Try to extract category from path or filename
    path_parts = [p.lower() for p in image_path.parts]
    
    # Common fashion categories
    categories = {
        'dress': 'a fashion photograph of a dress',
        'top': 'a fashion photograph of a top',
        'shirt': 'a fashion photograph of a shirt',
        'blouse': 'a fashion photograph of a blouse',
        'sweater': 'a fashion photograph of a sweater',
        'jacket': 'a fashion photograph of a jacket',
        'coat': 'a fashion photograph of a coat',
        'pants': 'a fashion photograph of pants',
        'jeans': 'a fashion photograph of jeans',
        'shorts': 'a fashion photograph of shorts',
        'skirt': 'a fashion photograph of a skirt',
        'suit': 'a fashion photograph of a suit',
        'tee': 'a fashion photograph of a t-shirt',
        't-shirt': 'a fashion photograph of a t-shirt',
        'hoodie': 'a fashion photograph of a hoodie',
        'cardigan': 'a fashion photograph of a cardigan',
        'blazer': 'a fashion photograph of a blazer',
        'vest': 'a fashion photograph of a vest',
        'jumpsuit': 'a fashion photograph of a jumpsuit',
        'romper': 'a fashion photograph of a romper',
    }
    
    # Check path parts and filename for category
    for key, caption in categories.items():
        if any(key in part for part in path_parts) or key in filename:
            return caption
    
    # Default caption
    if category:
        return f"a fashion photograph of {category}"
    
    return "a high quality fashion photograph of clothing, professional product photo"


def process_image(
    image_path: Path,
    output_path: Path,
    size: int,
    maintain_aspect: bool = False
) -> bool:
    """
    Process a single image: resize and save.
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if maintain_aspect:
                # Resize maintaining aspect ratio, then center crop
                ratio = size / min(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                
                # Center crop
                left = (img.size[0] - size) // 2
                top = (img.size[1] - size) // 2
                img = img.crop((left, top, left + size, top + size))
            else:
                # Simple resize
                img = img.resize((size, size), Image.LANCZOS)
            
            # Save as high-quality JPEG
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, 'JPEG', quality=95)
            return True
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False


def prepare_projected_gan_dataset(
    images: list,
    output_dir: Path,
    size: int = 256,
    max_images: int = None
):
    """
    Prepare dataset for Projected GAN.
    Creates a flat directory of resized images.
    """
    print("\n" + "="*50)
    print(f" Preparing Projected GAN Dataset ({size}x{size})")
    print("="*50)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Limit number of images if specified
    if max_images and len(images) > max_images:
        print(f"Limiting to {max_images} images (from {len(images)})")
        images = random.sample(images, max_images)
    
    # Process images in parallel
    processed = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for i, img_path in enumerate(images):
            output_path = output_dir / f"img_{i:06d}.jpg"
            future = executor.submit(
                process_image, img_path, output_path, size, maintain_aspect=True
            )
            futures[future] = img_path
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            if future.result():
                processed += 1
            else:
                failed += 1
    
    print(f"\n[OK] Processed: {processed} images")
    if failed > 0:
        print(f"✗ Failed: {failed} images")
    
    return processed


def prepare_lora_dataset(
    images: list,
    output_dir: Path,
    size: int = 512,
    max_images: int = None
):
    """
    Prepare dataset for LoRA training.
    Creates image-caption pairs with metadata.json
    """
    print("\n" + "="*50)
    print(f" Preparing LoRA Dataset ({size}x{size})")
    print("="*50)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    # Limit number of images if specified
    if max_images and len(images) > max_images:
        print(f"Limiting to {max_images} images (from {len(images)})")
        images = random.sample(images, max_images)
    
    # Process images and generate captions
    metadata = []
    processed = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for i, img_path in enumerate(images):
            output_path = image_dir / f"img_{i:06d}.jpg"
            future = executor.submit(
                process_image, img_path, output_path, size, maintain_aspect=True
            )
            futures[future] = (img_path, output_path, i)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            img_path, output_path, idx = futures[future]
            if future.result():
                caption = get_fashion_caption(img_path)
                metadata.append({
                    "file_name": output_path.name,
                    "text": caption
                })
                processed += 1
    
    # Save metadata
    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')
    
    print(f"\n[OK] Processed: {processed} images")
    print(f"[OK] Metadata saved to: {metadata_path}")
    
    return processed


def find_images(directory: Path) -> list:
    """Find all image files in a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    images = []
    
    for path in directory.rglob('*'):
        if path.suffix.lower() in image_extensions:
            images.append(path)
    
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DeepFashion dataset for training"
    )
    
    parser.add_argument(
        '--gan-size', 
        type=int, 
        default=256,
        help='Image size for Projected GAN (default: 256)'
    )
    
    parser.add_argument(
        '--lora-size', 
        type=int, 
        default=512,
        help='Image size for LoRA training (default: 512)'
    )
    
    parser.add_argument(
        '--max-gan-images',
        type=int,
        default=5000,
        help='Maximum images for GAN training (default: 5000)'
    )
    
    parser.add_argument(
        '--max-lora-images',
        type=int,
        default=1000,
        help='Maximum images for LoRA training (default: 1000)'
    )
    
    parser.add_argument(
        '--source-dir',
        type=str,
        default=None,
        help='Source directory with images (default: data/raw)'
    )
    
    parser.add_argument(
        '--skip-gan',
        action='store_true',
        help='Skip Projected GAN dataset preparation'
    )
    
    parser.add_argument(
        '--skip-lora',
        action='store_true',
        help='Skip LoRA dataset preparation'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print(" DeepFashion Dataset Preparation")
    print("="*50)
    
    # Find source directory - ONLY use train_images for real fashion photos
    # Avoid segm/ and densepose/ which contain masks, not real images!
    if args.source_dir:
        source_dir = Path(args.source_dir)
    else:
        # Check for train_images first (real photos)
        train_images_dir = RAW_DIR / "datasets" / "train_images"
        if train_images_dir.exists():
            source_dir = train_images_dir
            print(f"\n[INFO] Using train_images folder (real fashion photos)")
        else:
            source_dir = RAW_DIR
    
    if not source_dir.exists():
        print(f"\nERROR: Source directory not found: {source_dir}")
        print("Please run download_deepfashion.py first, or specify --source-dir")
        sys.exit(1)
    
    # Find all images
    print(f"\nSearching for images in: {source_dir}")
    images = find_images(source_dir)
    
    if not images:
        print("ERROR: No images found!")
        print("Please check that your images are in the data/raw directory")
        sys.exit(1)
    
    print(f"Found {len(images)} images")
    
    # Shuffle for random selection
    random.shuffle(images)
    
    # Prepare Projected GAN dataset
    if not args.skip_gan:
        prepare_projected_gan_dataset(
            images,
            GAN_DATA_DIR,
            size=args.gan_size,
            max_images=args.max_gan_images
        )
    
    # Prepare LoRA dataset
    if not args.skip_lora:
        prepare_lora_dataset(
            images,
            LORA_DATA_DIR,
            size=args.lora_size,
            max_images=args.max_lora_images
        )
    
    print("\n" + "="*50)
    print(" Dataset Preparation Complete!")
    print("="*50)
    print(f"\nProjected GAN data: {GAN_DATA_DIR}")
    print(f"LoRA data: {LORA_DATA_DIR}")
    print("\nNext steps:")
    print("  1. python projected_gan/train.py")
    print("  2. python stable_diffusion_lora/train_lora.py")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
