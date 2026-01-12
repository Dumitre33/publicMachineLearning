"""
Stable Diffusion LoRA Image Generation
======================================

Generate fashion images using fine-tuned LoRA weights.

Usage:
    python stable_diffusion_lora/generate.py --lora-path outputs/lora/checkpoint-final
    python stable_diffusion_lora/generate.py --lora-path outputs/lora/checkpoint-final --prompt "a red dress"
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel


# Default fashion prompts for generation
DEFAULT_PROMPTS = [
    "a high quality fashion photograph of an elegant black dress, professional product photo",
    "a fashion photograph of a casual white t-shirt, clean background, studio lighting",
    "a professional product photo of blue jeans, fashion photography",
    "a fashion photograph of a red evening gown, luxury fashion, studio shot",
    "a product photo of a leather jacket, fashion photography, professional lighting",
    "a fashion photograph of a summer floral dress, bright colors, studio photo",
    "a professional fashion photo of a business suit, formal wear, clean background",
    "a fashion photograph of a cozy sweater, autumn fashion, studio lighting",
]


def load_pipeline(
    lora_path: str,
    base_model: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda",
) -> StableDiffusionPipeline:
    """Load Stable Diffusion pipeline with LoRA weights."""
    
    print(f"Loading base model: {base_model}")
    
    # Load base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    
    # Use faster scheduler
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    
    # Load LoRA weights
    print(f"Loading LoRA weights from: {lora_path}")
    pipeline.unet = PeftModel.from_pretrained(
        pipeline.unet,
        lora_path,
    )
    
    # Move to device
    pipeline = pipeline.to(device)
    
    # Enable memory optimizations
    try:
        pipeline.enable_xformers_memory_efficient_attention()
        print("✓ xformers memory efficient attention enabled")
    except Exception as e:
        print(f"xformers not available: {e}")
    
    return pipeline


@torch.no_grad()
def generate_images(
    pipeline: StableDiffusionPipeline,
    prompts: List[str],
    num_images_per_prompt: int = 1,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    negative_prompt: str = "low quality, blurry, distorted, deformed, ugly",
    seed: int = None,
) -> List[Image.Image]:
    """Generate images from prompts."""
    
    all_images = []
    
    # Set seed if provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    for prompt in tqdm(prompts, desc="Generating"):
        images = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images
        
        all_images.extend(images)
    
    return all_images


def save_images(
    images: List[Image.Image],
    output_dir: Path,
    prompts: List[str] = None,
):
    """Save images with optional prompt metadata."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img in enumerate(images):
        filename = f"generated_{i:04d}.png"
        img.save(output_dir / filename)
    
    # Save prompts if provided
    if prompts:
        with open(output_dir / "prompts.txt", "w") as f:
            for i, prompt in enumerate(prompts):
                f.write(f"{i}: {prompt}\n")
    
    print(f"Saved {len(images)} images to {output_dir}")


def create_grid(images: List[Image.Image], cols: int = 4) -> Image.Image:
    """Create a grid image from multiple images."""
    
    n = len(images)
    rows = (n + cols - 1) // cols
    
    w, h = images[0].size
    grid = Image.new('RGB', (cols * w, rows * h), color='white')
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        grid.paste(img, (col * w, row * h))
    
    return grid


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with fine-tuned Stable Diffusion LoRA"
    )
    
    parser.add_argument(
        '--lora-path',
        type=str,
        required=True,
        help='Path to LoRA checkpoint'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Single prompt for generation'
    )
    
    parser.add_argument(
        '--prompts-file',
        type=str,
        default=None,
        help='File with prompts (one per line)'
    )
    
    parser.add_argument(
        '--num-images',
        type=int,
        default=1,
        help='Number of images per prompt'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=30,
        help='Number of inference steps'
    )
    
    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=7.5,
        help='Guidance scale (higher = more prompt adherence)'
    )
    
    parser.add_argument(
        '--negative-prompt',
        type=str,
        default="low quality, blurry, distorted, deformed, ugly, bad anatomy",
        help='Negative prompt'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory'
    )
    
    parser.add_argument(
        '--grid',
        action='store_true',
        help='Save as single grid image'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--use-defaults',
        action='store_true',
        help='Use default fashion prompts'
    )
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load pipeline
    pipeline = load_pipeline(args.lora_path, device=device)
    
    # Collect prompts
    prompts = []
    
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.use_defaults:
        prompts = DEFAULT_PROMPTS
    else:
        # Interactive mode
        print("\nEnter prompts (empty line to finish):")
        while True:
            prompt = input("> ").strip()
            if not prompt:
                break
            prompts.append(prompt)
        
        if not prompts:
            prompts = DEFAULT_PROMPTS[:4]
            print(f"Using default prompts: {len(prompts)}")
    
    print(f"\nGenerating {len(prompts)} x {args.num_images} = {len(prompts) * args.num_images} images")
    
    # Generate images
    images = generate_images(
        pipeline,
        prompts,
        num_images_per_prompt=args.num_images,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
    )
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PROJECT_ROOT / 'outputs' / 'lora' / 'generated' / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save images
    if args.grid:
        grid = create_grid(images)
        grid.save(output_dir / 'grid.png')
        print(f"Saved grid to {output_dir / 'grid.png'}")
    else:
        save_images(images, output_dir, prompts)
    
    print("\n✓ Generation complete!")


if __name__ == "__main__":
    main()
