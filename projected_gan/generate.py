"""
Projected GAN Image Generation
==============================

Generate fashion images using a trained Projected GAN model.

Usage:
    python projected_gan/generate.py --checkpoint outputs/projected_gan/checkpoint_final.pt
    python projected_gan/generate.py --checkpoint outputs/projected_gan/checkpoint_final.pt --num-samples 64
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from projected_gan.model import Generator


def load_generator(checkpoint_path: str, device: str = 'cuda') -> Generator:
    """Load generator from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    
    G = Generator(
        z_dim=config['model']['latent_dim'],
        img_size=config['model']['img_size'],
    ).to(device)
    
    G.load_state_dict(checkpoint['G_state_dict'])
    G.eval()
    
    return G, config


@torch.no_grad()
def generate_images(
    G: Generator,
    num_samples: int = 16,
    truncation: float = 0.7,
    batch_size: int = 16,
    device: str = 'cuda',
) -> torch.Tensor:
    """Generate multiple images."""
    all_images = []
    
    for i in tqdm(range(0, num_samples, batch_size), desc="Generating"):
        current_batch = min(batch_size, num_samples - i)
        
        z = torch.randn(current_batch, G.z_dim, device=device)
        
        if truncation < 1.0:
            z = z * truncation
        
        images = G(z)
        images = (images + 1) / 2  # [-1, 1] -> [0, 1]
        all_images.append(images.cpu())
    
    return torch.cat(all_images, dim=0)


@torch.no_grad()
def interpolate(
    G: Generator,
    num_steps: int = 10,
    num_pairs: int = 4,
    device: str = 'cuda',
) -> torch.Tensor:
    """Create interpolation between random latents."""
    all_frames = []
    
    for _ in range(num_pairs):
        z1 = torch.randn(1, G.z_dim, device=device)
        z2 = torch.randn(1, G.z_dim, device=device)
        
        for t in np.linspace(0, 1, num_steps):
            z = (1 - t) * z1 + t * z2
            img = G(z)
            img = (img + 1) / 2
            all_frames.append(img.cpu())
    
    return torch.cat(all_frames, dim=0)


def save_grid(images: torch.Tensor, path: str, nrow: int = 4):
    """Save images as a grid."""
    grid = make_grid(images, nrow=nrow, padding=2)
    save_image(grid, path)


def save_individual(images: torch.Tensor, output_dir: Path):
    """Save images individually."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img in enumerate(images):
        img_pil = Image.fromarray(
            (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )
        img_pil.save(output_dir / f'generated_{i:04d}.jpg', quality=95)


def main():
    parser = argparse.ArgumentParser(description="Generate images with Projected GAN")
    
    parser.add_argument(
        '--checkpoint', 
        type=str, 
        required=True,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--num-samples', 
        type=int, 
        default=16,
        help='Number of images to generate'
    )
    
    parser.add_argument(
        '--truncation', 
        type=float, 
        default=0.7,
        help='Truncation factor (0-1, lower = higher quality but less diversity)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Output path (default: outputs/projected_gan/generated/)'
    )
    
    parser.add_argument(
        '--grid', 
        action='store_true',
        help='Save as single grid image'
    )
    
    parser.add_argument(
        '--interpolate', 
        action='store_true',
        help='Generate interpolation between random latents'
    )
    
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    G, config = load_generator(args.checkpoint, device)
    print(f"Model loaded: {config['model']['img_size']}x{config['model']['img_size']}")
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = PROJECT_ROOT / 'outputs' / 'projected_gan' / 'generated' / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate
    if args.interpolate:
        print("Generating interpolation...")
        images = interpolate(G, num_pairs=4, num_steps=10, device=device)
        save_grid(images, output_dir / 'interpolation.png', nrow=10)
        print(f"Saved interpolation to {output_dir / 'interpolation.png'}")
    else:
        print(f"Generating {args.num_samples} images (truncation={args.truncation})...")
        images = generate_images(
            G, 
            num_samples=args.num_samples,
            truncation=args.truncation,
            device=device,
        )
        
        if args.grid:
            nrow = int(np.sqrt(args.num_samples))
            save_grid(images, output_dir / 'grid.png', nrow=nrow)
            print(f"Saved grid to {output_dir / 'grid.png'}")
        else:
            save_individual(images, output_dir)
            print(f"Saved {args.num_samples} images to {output_dir}")
    
    print("\n✓ Generation complete!")


if __name__ == "__main__":
    main()
